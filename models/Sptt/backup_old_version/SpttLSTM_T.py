import os
import torch
import sys
import torch.nn as nn
import math
from global_param import global_param
import time
from utils.GradientAccumulator_Stream import GradientAccumulator_hidden

accumulator = GradientAccumulator_hidden()

def matrix_product_magnitude_estimation(A, B, C, 
                                        A_norm, B_norm, C_norm, 
                                        sample_size=100, epsilon=1e-10):
    """
    使用批处理随机抽样估计三矩阵乘积量级
    
    参数:
        A: 形状为 mxn 的矩阵
        B: 形状为 nxn 的对角矩阵或对角线元素向量(长度为n)
        C: 形状为 dxn 的矩阵
        A_norm, B_norm, C_norm: 归一化矩阵
        sample_size: 随机向量的数量
        epsilon: 数值稳定性参数
    
    返回:
        三矩阵乘积的范数估计值
    """
    sample_size = global_param['hutchinson_size']
    print("sample_size: ", sample_size)
    
    device = A.device
    d = C.shape[0]
    
    # 创建一次性批量随机向量 [d, sample_size]
    V = torch.randn(d, sample_size, device=device)
    # 对每个向量进行归一化，保持单位长度
    V_norms = torch.norm(V, dim=0, keepdim=True)
    V = V / V_norms  # [d, sample_size]
    
    # 计算 C^T @ V，结果形状为 [n, sample_size]
    CV = C.t() @ V
    
    # 应用对角矩阵 B（按元素乘法）
    # 扩展 B 的维度以适应批处理
    B_expanded = B.unsqueeze(1)  # [n, 1]
    BCV = CV * B_expanded  # [n, sample_size]
    
    # 计算 A @ (B×C^T×V)
    result = A @ BCV  # [m, sample_size]
    
    # 使用归一化矩阵计算
    Cnorm_V = C_norm.t() @ V
    BCnorm_V = Cnorm_V * B_norm.unsqueeze(1)
    result_norm = A_norm @ BCnorm_V
    
    # 计算每个样本的范数平方
    norm_squared = torch.sum(result * result, dim=0)  # [sample_size]
    norm_squared_norm = torch.sum(result_norm * result_norm, dim=0)
    
    # 计算原始量级和新量级
    original_product = torch.sqrt(torch.mean(norm_squared))
    new_product = torch.sqrt(torch.mean(norm_squared_norm))
    
    # 计算缩放因子
    scaling_factor = (original_product / (new_product + epsilon)) ** (1/3)
    
    return scaling_factor

# V1版本相比于传统的StittLSTM优化了QR分解的效率,但其本质不变， 以及对一些处理预先分配内存。
# V1版本结果很令人兴奋，结果和StittLSTM一样，但更快了。
# 探究采样行数对准确性，收敛速度，传播时间的影响。

def normalize_to_standard_normal(matrix):
    """
    将矩阵元素缩放为服从标准正态分布(均值0，标准差1)
    
    参数:
        matrix: 输入矩阵
    
    返回:
        缩放后的矩阵
    """
    # 计算当前矩阵的统计信息
    current_mean = torch.mean(matrix)
    current_std = torch.std(matrix)
    
    # 首先中心化数据(减去均值)
    centered = matrix - current_mean
    
    # 然后缩放到单位标准差
    if current_std > 0:  # 避免除以零
        normalized = centered / current_std
    else:
        # 如果标准差为零，生成随机标准正态噪声
        normalized = torch.randn_like(matrix)
    
    return normalized

def consistent_hutchinson_normalize_matrices(X_matrix, Sigma_diag, Delta_matrix, 
                                     preserve_product=True):
    """
    使用一致掩码的矩阵归一化函数，不对奇异值矩阵应用掩码
    
    参数:
        X_matrix: 左奇异矩阵
        Sigma_diag: 奇异值向量
        Delta_matrix: 右奇异矩阵
        preserve_product: 是否保持三个矩阵乘积的量级
        dropout_rate_X, dropout_rate_D: 失活率
        trials: 重复试验次数
        epsilon: 数值稳定性小量
    
    返回:
        标准正态化后的三个矩阵
    """
    # 对矩阵进行标准化处理
    X_normalized = normalize_to_standard_normal(X_matrix)
    Sigma_normalized = torch.abs(normalize_to_standard_normal(Sigma_diag))
    Delta_normalized = normalize_to_standard_normal(Delta_matrix)
    
    # 如果需要保持乘积不变
    if preserve_product:
        # 使用一致掩码估计原始乘积和归一化后乘积
        scaling_factor = matrix_product_magnitude_estimation(
            X_matrix, Sigma_diag, Delta_matrix,
            X_normalized, Sigma_normalized, Delta_normalized,
        )
        
        # 对所有三个矩阵应用缩放因子
        X_normalized = X_normalized * scaling_factor
        Sigma_normalized = Sigma_normalized * scaling_factor
        Delta_normalized = Delta_normalized * scaling_factor
    
    return X_normalized, Sigma_normalized, Delta_normalized

# 使用更高效的方式更新矩阵形状
def QR_matrix(X_matrix, target_shape):
    Q, R = qr_with_non_negative_diagonal(X_matrix)
    
    # 使用预分配和高效索引
    if Q.shape != target_shape:
        result = torch.zeros(target_shape, device=X_matrix.device, dtype=X_matrix.dtype)
        result[:min(Q.shape[0], target_shape[0]), :min(Q.shape[1], target_shape[1])] = \
            Q[:min(Q.shape[0], target_shape[0]), :min(Q.shape[1], target_shape[1])]
        return result
    return Q

    
def qr_with_non_negative_diagonal(matrix):
    # 进行QR分解
    Q, R = torch.linalg.qr(matrix)
    
    # 获取 R 矩阵对角线元素的符号
    diagonal_sign = torch.sign(torch.diag(R))
    
    # 避免对角线元素为零的情况
    diagonal_sign[diagonal_sign == 0] = 1
    
    # 调整 Q 矩阵的列符号
    Q = Q * diagonal_sign.unsqueeze(0)

    return Q, R

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.seq_len = config.seq_len
        print("seq_len: ", self.seq_len)
        self.hidden_dim = config.hidden_dim
        self.device = config.gpu_type
        self.krank = config.krank
        self.task_name = config.task_name
        self.time_block_num = config.time_block_num
        # 如果是多变量预测多变量的问题，全连接层的输出尺寸应该是 特征维度 X 预测长度
        if self.task_name == 'long_term_forecast':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim
        
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")

    def forward(self, x):
        final_hidden = self.lstm(x)
        fc_out = self.fc(final_hidden)
        # print("fc_out shape: ", fc_out.shape)
        # fc_out shape [64, 672]
        if self.task_name == 'long_term_forecast':
            y = fc_out.view(-1, self.output_size, self.feature_dim)
        elif self.task_name == 'classification':
            y = fc_out
        elif self.task_name == 'imputation':
            y = fc_out.view(-1, self.output_size, self.feature_dim)
        elif self.task_name == 'anomaly_detection':
            y = fc_out.view(-1, self.output_size, self.feature_dim)
        # print ("y shape: ", y.shape)
        # y shape [64, 96, 7]
        return y
    
class CustomLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, time_block_num, device):
        super(CustomLSTM, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.krank = krank
        self.time_block_num = time_block_num
        self.device = device
        
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, feature_dim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))
        
        self.reset_parameters()
        self.init_stitt_parameters()
        
        accumulator.save_sbtpca_parameter(self.X_matrix_ih, self.Sigma_ih, self.Sigma_matrix_ih, self.Delta_matrix_ih, 
                                          self.X_matrix_hh, self.Sigma_hh, self.Sigma_matrix_hh, self.Delta_matrix_hh)
    
    def reset_parameters(self):
        # 定义标准差
        std = 1.0 / math.sqrt(self.hidden_dim)
        # 初始化参数
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
            
    def init_stitt_parameters(self):
        self.X_matrix_ih = torch.randn(self.feature_dim, self.krank).to(self.device)
        self.Sigma_ih = torch.randn(self.krank).to(self.device)
        self.Sigma_matrix_ih = torch.diag(self.Sigma_ih).to(self.device)
        self.Delta_matrix_ih = torch.randn(4 * self.hidden_dim, self.krank).to(self.device)
        
        self.X_matrix_hh = torch.randn(self.hidden_dim, self.krank).to(self.device)
        self.Sigma_hh = torch.randn(self.krank).to(self.device)
        self.Sigma_matrix_hh = torch.diag(self.Sigma_hh).to(self.device)
        self.Delta_matrix_hh = torch.randn(4 * self.hidden_dim, self.krank).to(self.device)

    def forward(self, x):
        accumulator.reset()
        global_param['flag'] = True
        # global_param['comporession_finished'] = False
        global_param['compression_finished'] = False
        
        # 获取输入序列的第一个尺度batch_size，进行隐藏状态和细胞状态的初始化
        num_samples = x.size(0)
        seq_len = x.size(1)
        
        # 在RNN做时间序列预测中，要对每个样本初始化一个隐藏状态和细胞状态
        hx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        cx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        
        # 总的元素数
        T = num_samples * seq_len
        
        global_param['total_time_block_size'] = T
        
        for i in range(x.size(1)):
            inputs = x[:, i, :]
            hx, cx = LSTMCellFunction.apply(
                inputs, hx, cx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.time_block_num
            )

        return hx
    
class LSTMCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hx, cx, w_ih, w_hh, b_ih, b_hh, time_block_num):
        gates = (torch.mm(inputs, w_ih.t()) + b_ih + torch.mm(hx, w_hh.t()) + b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        ctx.save_for_backward(inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh)
        ctx.time_block_num = time_block_num
        
        # 返回一个零时间，真正的时间将在backward中测量。
        return hy, cy
    
    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh = ctx.saved_tensors
        time_block_num = ctx.time_block_num
        
        # 初始化梯度
        grad_inputs = grad_hx = grad_cx = grad_w_ih = grad_w_hh = grad_b_ih = grad_b_hh = None
        
        if not global_param['compression_finished'] and global_param['flag']:
            # 计算梯度
            grad_outgate = grad_hy * torch.tanh(cy) * outgate * (1 - outgate)
            grad_cy = grad_hy * outgate * (1 - torch.tanh(cy) ** 2) + grad_cy
            grad_ingate = grad_cy * cellgate * ingate * (1 - ingate)
            grad_cellgate = grad_cy * ingate * (1 - cellgate ** 2)
            grad_forgetgate = grad_cy * cx * forgetgate * (1 - forgetgate)
            
            # if global_param['flag']:
            # 保存stitt所用的参数
            delta = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1)
            accumulator.accumulate(inputs, hx, delta)
            
            # 计算输入和隐藏状态的梯度
            grad_inputs = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_ih)
            grad_hx = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_hh)
            
            # 计算偏置的梯度
            grad_b_ih = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
            grad_b_hh = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
            
            # 计算上一个单元状态的梯度
            grad_cx = grad_cy * forgetgate
        
        if global_param['compression_finished'] and global_param['flag']:
            grad_w_ih = torch.zeros_like(w_ih)
            grad_w_hh = torch.zeros_like(w_hh)
            
            global_param['flag'] = False
            
            # 取stitt所需参数
            X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih, X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh = accumulator.deliver_sbtpca_parameter()
            input_cat, hx_cat, delta_cat = accumulator.get_concatenated_gradients()
            
            T = len(input_cat)
            t = T // time_block_num
            # print("T // t", T // t)
            inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
            inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)
            
            start_calculate_gradients = time.time()
            with torch.no_grad():
                for i in range(1, T // t + 1):
                    # print("运行")
                    start_idx = (i - 1) * t
                    activation_input = input_cat[start_idx:start_idx + t] # 半开区间，不包括结束索引
                    # activation_input = F.dropout(activation_input, p=(1 / (i + 1)), training=True) # 0.2 测过了效果不好
                    activation_hx = hx_cat[start_idx:start_idx + t]
                    # activation_hx = F.dropout(activation_hx, p=(1 / (i + 1)), training=True)
                    delta = delta_cat[start_idx:start_idx + t]
                    # delta = F.dropout(delta, p=(1 / (i + 1)), training=True)
                    
                    scale_factor_right_ih = delta @ Delta_matrix_ih / t
                    scale_factor_left_ih = activation_input @ X_matrix_ih / t
                    
                    scale_factor_right_hh = delta @ Delta_matrix_hh / t
                    scale_factor_left_hh = activation_hx @ X_matrix_hh / t
                    
                    # 使用预计算的系数提高效率
                    i_factor = i / (i + 1)
                    update_factor = 1 / (i + 1)
                    
                    # 更新X和Delta矩阵
                    X_ih_update = activation_input.t() @ scale_factor_right_ih @ inv_Sigma_matrix_ih
                    X_matrix_ih = i_factor * X_matrix_ih + update_factor * X_ih_update
                    
                    Delta_ih_update = delta.t() @ scale_factor_left_ih @ inv_Sigma_matrix_ih
                    Delta_matrix_ih = i_factor * Delta_matrix_ih + update_factor * Delta_ih_update

                    X_hh_update = activation_hx.t() @ scale_factor_right_hh @ inv_Sigma_matrix_hh
                    X_matrix_hh = i_factor * X_matrix_hh + update_factor * X_hh_update

                    Delta_hh_update = delta.t() @ scale_factor_left_hh @ inv_Sigma_matrix_hh
                    Delta_matrix_hh = i_factor * Delta_matrix_hh + update_factor * Delta_hh_update
                
                    X_matrix_ih = QR_matrix(X_matrix_ih, X_matrix_ih.shape)
                    Delta_matrix_ih = QR_matrix(Delta_matrix_ih, Delta_matrix_ih.shape)
                    
                    X_matrix_hh = QR_matrix(X_matrix_hh, X_matrix_hh.shape)
                    Delta_matrix_hh = QR_matrix(Delta_matrix_hh, Delta_matrix_hh.shape)
                    
                    # 批量计算Sigma更新
                    Sigma_ih_product = torch.sum(((delta @ Delta_matrix_ih) * (activation_input @ X_matrix_ih)) / ((i + 1) * t), dim=0)
                    Sigma_ih = i_factor * Sigma_ih + Sigma_ih_product
                    
                    Sigma_hh_product = torch.sum(((delta @ Delta_matrix_hh) * (activation_hx @ X_matrix_hh)) / ((i + 1) * t), dim=0)
                    Sigma_hh = i_factor * Sigma_hh + Sigma_hh_product
                    
                    # 在计算完成后、更新梯度前应用对数缩放
                    Sigma_matrix_ih = torch.diag(Sigma_ih)
                    Sigma_matrix_hh = torch.diag(Sigma_hh)
                    
                    # 更新逆矩阵
                    inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
                    inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)
                    
            X_matrix_ih_Gauss, Sigma_ih_Gauss, Delta_matrix_ih_Gauss = consistent_hutchinson_normalize_matrices(X_matrix_ih, Sigma_ih, Delta_matrix_ih, preserve_product=True)
            X_matrix_hh_Gauss, Sigma_hh_Gauss, Delta_matrix_hh_Gauss = consistent_hutchinson_normalize_matrices(X_matrix_hh, Sigma_hh, Delta_matrix_hh, preserve_product=True)                
            
            Sigma_matrix_ih_Gauss = torch.diag(Sigma_ih_Gauss)
            Sigma_matrix_hh_Gauss = torch.diag(Sigma_hh_Gauss)
            
            grad_w_ih = (X_matrix_ih_Gauss @ Sigma_matrix_ih_Gauss @ Delta_matrix_ih_Gauss.t()).t()
            grad_w_hh = (X_matrix_hh_Gauss @ Sigma_matrix_hh_Gauss @ Delta_matrix_hh_Gauss.t()).t()
            
            end_calculate_gradients = time.time()
            global_param['time_for_batch'] = end_calculate_gradients - start_calculate_gradients
            
            accumulator.save_sbtpca_parameter(X_matrix_ih_Gauss, Sigma_ih_Gauss, Sigma_matrix_ih_Gauss, Delta_matrix_ih_Gauss, 
                                              X_matrix_hh_Gauss, Sigma_hh_Gauss, Sigma_matrix_hh_Gauss, Delta_matrix_hh_Gauss)
            
        return grad_inputs, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None