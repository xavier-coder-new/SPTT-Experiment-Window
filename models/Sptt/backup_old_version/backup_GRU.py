import os
import torch
import sys
import torch.nn as nn
import math
from global_param import global_param
import time
from utils.GradientAccumulator import GradientAccumulator
import numpy as np

accumulator = GradientAccumulator()

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
    # print("sample_size: ", sample_size)
    
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
    # scaling_factor = (original_product / (new_product + epsilon)) ** (1/3)
    
    scaling_factor = (original_product / (new_product + epsilon)) ** (1)
    
    return scaling_factor


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
    Sigma_normalized = torch.abs(normalize_to_standard_normal(Sigma_diag))
    
    # 如果需要保持乘积不变
    if preserve_product:
        # 使用一致掩码估计原始乘积和归一化后乘积
        scaling_factor = matrix_product_magnitude_estimation(
            X_matrix, Sigma_diag, Delta_matrix,
            X_matrix, Sigma_normalized, Delta_matrix,
        )
        
        # 应用缩放因子
        Sigma_normalized = Sigma_normalized * scaling_factor
    
    return Sigma_normalized

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
    
    # 调整 Q 矩阵的列符号和Q = Q * diagonal_sign得到的结果一样。
    Q1 = Q * diagonal_sign.unsqueeze(0)
    # Q2 = Q * diagonal_sign
    # print("Are results equal?", torch.allclose(Q1, Q2, rtol=1e-5))
    
    return Q1, R

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
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device,config)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")

    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        loss = self.gru(x, target, criterion, model_optim, inp=inp, mask=mask)
        # fc_out = self.fc(final_hidden)
        # # print("fc_out shape: ", fc_out.shape)
        # # fc_out shape [64, 672]
        # if self.task_name == 'long_term_forecast':
        #     y = fc_out.view(-1, self.output_size, self.feature_dim)
        # elif self.task_name == 'classification':
        #     y = fc_out
        # elif self.task_name == 'imputation':
        #     y = fc_out.view(-1, self.output_size, self.feature_dim)
        # elif self.task_name == 'anomaly_detection':
        #     y = fc_out.view(-1, self.output_size, self.feature_dim)
        # # print ("y shape: ", y.shape)
        # # y shape [64, 96, 7]
        return loss
    
class CustomGRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, time_block_num, device, config):
        super(CustomGRU, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.krank = krank
        self.time_block_num = time_block_num
        self.device = device
        self.task_name = config.task_name
        self.output_size = config.pred_len
        self.feature_dim = config.feature_dim
        self.output_size_anomaly = config.seq_len
        self.config = config
        
        self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, feature_dim))
        self.w_hh = nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(3 * hidden_dim))
        
        if self.task_name == 'classification':
            self.fc = nn.Linear(self.hidden_dim, self.config.num_classes)
        elif self.task_name == 'long_term_forecast':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size_anomaly)
        elif self.task_name == 'imputation':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size_anomaly)
        
        self.reset_parameters()
        self.init_stitt_parameters()
        
        accumulator.save_sptt_parameter(self.X_matrix_ih, self.Sigma_ih, self.Sigma_matrix_ih, self.Delta_matrix_ih, 
                                          self.X_matrix_hh, self.Sigma_hh, self.Sigma_matrix_hh, self.Delta_matrix_hh)
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim) # 定义标准差

        for weight in self.parameters(): # 遍历模型的所有参数
            weight.data.uniform_(-std, std) # 使用均匀分布初始化参数   
    
    def init_stitt_parameters(self):
        self.X_matrix_ih = torch.randn(self.feature_dim, self.krank).to(self.device)
        self.Sigma_ih = torch.randn(self.krank).to(self.device)
        self.Sigma_matrix_ih = torch.diag(self.Sigma_ih).to(self.device)
        self.Delta_matrix_ih = torch.randn( 3 * self.hidden_dim, self.krank).to(self.device)
        
        self.X_matrix_hh = torch.randn(self.hidden_dim, self.krank).to(self.device)
        self.Sigma_hh = torch.randn(self.krank).to(self.device)
        self.Sigma_matrix_hh = torch.diag(self.Sigma_hh).to(self.device)
        self.Delta_matrix_hh = torch.randn(3 * self.hidden_dim, self.krank).to(self.device)
        
    def forward(self, x, targets=None, criterion=None, model_optim=None, inp=None, mask=None):
        global_param['time_for_batch'] = 0.0
        
        if self.task_name == 'imputation':
            x_new = x
            x = inp
        
        # 获取输入序列的第一个尺度batch_size，进行隐藏状态和细胞状态的初始化
        num_samples = x.size(0)
        seq_len = x.size(1)
        # 程序测试没问题-StittGRU
        # global_param['iter'] = 0
        
        # 在RNN做时间序列预测中，要对每个样本初始化一个隐藏状态
        hx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        
        # 划分时间块的索引
        indices = np.linspace(0, seq_len, self.time_block_num + 1, dtype=int)
        
        sum_loss = 0.0
        
        if model_optim is not None:
            chunk_block = 0
            for i in range(self.time_block_num):
                accumulator.reset()
                chunk_start = indices[i]
                chunk_end = indices[i + 1]
                chunk_block += 1
                # 判断是否是最后一个chunk.
                if i == self.time_block_num - 1:
                    is_last_chunk = True
                else:
                    is_last_chunk = False
                
                # 截断梯度流，但保持状态连续性
                if chunk_start > 0:
                    # 保留数值前向继续，但断开反向传播图,，这样确保每个时间块之间状态连续但梯度不连续。
                    hx.detach_()
                
                # 处理当前chunk的时间步
                time_step = 0
                for i in range(chunk_start, chunk_end):
                    # 这是一个三维的张量，表示取x的全部batch，第一个时间步的全部特征维度。
                    # 这样数据的话每次就是
                    # [1 1 1 1
                    # 2 2 2 2
                    # 3 3 3 3]
                    time_step += 1
                    # 只有反向传播到第一个时间步的时候，表示当前时间块处理完毕，再进行梯度更新
                    inputs = x[:, i, :]
                    hx = GRUCellFunction.apply(
                            inputs, hx , self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step, chunk_block
                    )

                    
                # global_param['compression_finished'] = True
                
                fc_out = self.fc(hx)
                
                # 计算损失(训练)
                if self.task_name == "long_term_forecast":
                    # print("fc_out", fc_out.shape)
                    y = fc_out.view(-1, self.output_size, self.feature_dim)
                    f_dim = -1 if self.config.features == 'MS' else 0
                    outputs = y[:, -self.config.pred_len:, f_dim:]
                    targets = targets[:, -self.config.pred_len:, f_dim:]
                    loss = criterion(outputs, targets)
                elif self.task_name == 'classification':
                    outputs = fc_out
                    loss = criterion(outputs, targets.long().squeeze(-1))
                elif self.task_name == 'anomaly_detection':
                    # print("fc_out", fc_out.shape)
                    # print("进来了进来了！！！！！！！！！！！！！！！！！")
                    y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
                    f_dim = -1 if self.config.features == 'MS' else 0
                    outputs = y[:, :, f_dim:]
                    loss = criterion(outputs, x)
                elif self.task_name == 'imputation':      
                    y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
                    f_dim = -1 if self.config.features == 'MS' else 0
                    outputs = y[:, :, f_dim:]
                    batch_x = x_new[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]
                    
                    loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                
                sum_loss += loss.item()    
                    
                if not is_last_chunk:
                    # 如果不是最后一个chunk进行反向传播更新梯度
                    model_optim.zero_grad()
                    loss.backward()
                    model_optim.step()

            # 如果是最后一个last_chunk，直接返回loss，在主程序反向传播，更新梯度和权重 
            if self.task_name == 'classification':
                return outputs, loss, sum_loss
            else:
                return loss, sum_loss
            
        elif model_optim is None and criterion is not None:
            # 进行验证
            time_step = 0
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx = GRUCellFunction.apply(
                        inputs, hx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step
                )

            fc_out = self.fc(hx)
            
            if self.task_name == "long_term_forecast":
                y = fc_out.view(-1, self.output_size, self.feature_dim)
                f_dim = -1 if self.config.features == 'MS' else 0
                outputs = y[:, -self.config.pred_len:, f_dim:]
                targets = targets[:, -self.config.pred_len:, f_dim:]
                
                pred = outputs.detach().cpu()
                true = targets.detach().cpu()
                loss = criterion(pred, true)
            elif self.task_name == 'classification':
                outputs = fc_out
                # pred = outputs.detach().cpu()
                # loss = criterion(pred, targets.long().squeeze().cpu())
            elif self.task_name == 'anomaly_detection':
                y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
                f_dim = -1 if self.config.features == 'MS' else 0
                outputs = y[:, :, f_dim:]
                
                pred = outputs.detach().cpu()
                true = x.detach().cpu()
                loss = criterion(pred, true)
            elif self.task_name == 'imputation':
                y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
                f_dim = -1 if self.config.features == 'MS' else 0
                outputs = y[:, :, f_dim:]
                batch_x = x_new[:, :, f_dim:]
                mask = mask[:, :, f_dim:]
                
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()
                
                loss = criterion(pred[mask == 0], true[mask == 0])
            
            if self.task_name == 'classification':
                return outputs
            else:
                return loss
            
        elif model_optim is None and criterion is None:
            # 进行测试
            accumulator.reset()
            # num_samples = x.size(0)
            # seq_len = x.size(1)
            hx = torch.zeros(num_samples, self.hidden_dim).to(self.device)

            
            time_step = 1
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx = GRUCellFunction.apply(
                    inputs, hx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step
                )

            fc_out = self.fc(hx)
            
            if self.task_name == 'long_term_forecast':
                y = fc_out.view(-1, self.output_size, self.feature_dim)
            elif self.task_name == 'classification':
                y = fc_out
            elif self.task_name == 'anomaly_detection':
                y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
            elif self.task_name == 'imputation':
                y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
                
            return y   
        # for i in range(x.size(1)):
        #     inputs = x[:, i, :]
        #     hx = GRUCellFunction.apply(
        #         inputs, hx , self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.time_block_num
        # )

class GRUCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hidden_state, w_ih, w_hh, b_ih, b_hh, time_step=None, chunk_block=None):
        # 拆分权重矩阵和偏置向量
        w_ir, w_iz, w_in = w_ih.chunk(3, 0)
        w_hr, w_hz, w_hn = w_hh.chunk(3, 0)
        b_ir, b_iz, b_in = b_ih.chunk(3)
        b_hr, b_hz, b_hn = b_hh.chunk(3)
        
        # 计算重置门
        reset_gate = torch.sigmoid(
            torch.mm(input, w_ir.t()) + b_ir + torch.mm(hidden_state, w_hr.t()) + b_hr
        )
        
        # 计算更新门
        update_gate = torch.sigmoid(
            torch.mm(input, w_iz.t()) + b_iz + torch.mm(hidden_state, w_hz.t()) + b_hz
        )
        
        # 计算候选隐藏状态
        n = torch.tanh(
            torch.mm(input, w_in.t()) + b_in + reset_gate * (torch.mm(hidden_state, w_hn.t()) + b_hn)
        )
        
        # 计算新的隐藏状态
        hy = (1 - update_gate) * n + update_gate * hidden_state

        # 保存变量以供反向传播使用
        ctx.save_for_backward(input, hidden_state, w_ir, w_iz, w_in, w_hr, w_hz, w_hn,
                              reset_gate, update_gate, n, w_ih, w_hh, b_ir, b_iz, b_in, b_hr, b_hz, b_hn)
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block
        
        return hy
    
    @staticmethod
    def backward(ctx, grad_hy):
        
        input, hidden_state, w_ir, w_iz, w_in, w_hr, w_hz, w_hn, \
        reset_gate, update_gate, n, w_ih, w_hh, b_ir, b_iz, b_in, b_hr, b_hz, b_hn = ctx.saved_tensors
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block
        
        # 初始化梯度
        d_input = d_hidden_state = grad_w_ih = grad_w_hh = grad_b_ih = grad_b_hh = None
        
        # 计算更新门的梯度
        d_update_gate = (hidden_state - n) * grad_hy
        d_update_gate = d_update_gate * update_gate * (1 - update_gate)

        # 计算候选隐藏状态的梯度
        d_n = (1 - update_gate) * grad_hy
        d_n_tanh = d_n * (1 - n ** 2)

        # 计算重置门的梯度
        hidden_w_hn = torch.mm(hidden_state, w_hn.t()) + b_hn
        d_reset_gate = d_n_tanh * hidden_w_hn
        d_reset_gate = d_reset_gate * reset_gate * (1 - reset_gate)
        
        # 保存sptt所用的参数
        delta = torch.cat((d_reset_gate, d_update_gate, d_n_tanh), 1)
        # CustomPrint.custom_print()
        accumulator.accumulate(input, hidden_state, delta)
            
        # 计算输入和隐藏状态的梯度
        # 计算输入的梯度
        d_input = (
            torch.mm(d_reset_gate, w_ir) +
            torch.mm(d_update_gate, w_iz) +
            torch.mm(d_n_tanh, w_in)
        )

        # 计算前一个隐藏状态的梯度
        d_hidden_state = (
            grad_hy * update_gate +
            torch.mm(d_reset_gate, w_hr) +
            torch.mm(d_update_gate, w_hz) +
            reset_gate * torch.mm(d_n_tanh, w_hn)
        )
            
        # 计算 b_ih 的梯度
        grad_b_ir = d_reset_gate.sum(0)
        grad_b_iz = d_update_gate.sum(0)
        grad_b_in = d_n_tanh.sum(0)
        grad_b_ih = torch.cat((grad_b_ir, grad_b_iz, grad_b_in))

        # 计算 b_hh 的梯度
        grad_b_hr = d_reset_gate.sum(0)
        grad_b_hz = d_update_gate.sum(0)
        grad_b_hn = (d_n_tanh * reset_gate).sum(0)
        grad_b_hh = torch.cat((grad_b_hr, grad_b_hz, grad_b_hn))
            
        if time_step == 1:
            # print("!!!!!!!!!!!!!!!进入stitt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            grad_w_ih = torch.zeros_like(w_ih)
            grad_w_hh = torch.zeros_like(w_hh)
            
            global_param['flag'] = False
            
            # 取stitt所需参数
            X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih, X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh = accumulator.deliver_sptt_parameter()
            activation_input, activation_hx, delta = accumulator.get_concatenated_gradients()
            
            t = len(activation_input)
            i = chunk_block
            # i = t / 4
            # print("T // t", T // t)
            inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
            inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)
            
            start_calculate_gradients = time.time()
            with torch.no_grad():
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
                    
            Sigma_ih_Gauss = consistent_hutchinson_normalize_matrices(X_matrix_ih, Sigma_ih, Delta_matrix_ih, preserve_product=True)
            Sigma_hh_Gauss = consistent_hutchinson_normalize_matrices(X_matrix_hh, Sigma_hh, Delta_matrix_hh, preserve_product=True)                
            
            # print(Sigma_ih_Gauss)
            Sigma_matrix_ih_Gauss = torch.diag(Sigma_ih_Gauss)
            Sigma_matrix_hh_Gauss = torch.diag(Sigma_hh_Gauss)
            
            grad_w_ih = (X_matrix_ih @ Sigma_matrix_ih_Gauss @ Delta_matrix_ih.t()).t()
            grad_w_hh = (X_matrix_hh @ Sigma_matrix_hh_Gauss @ Delta_matrix_hh.t()).t()
            
            end_calculate_gradients = time.time()
            time_backward = end_calculate_gradients - start_calculate_gradients
            
            global_param['time_for_batch'] += time_backward
            
            
            accumulator.save_sptt_parameter(X_matrix_ih, Sigma_ih_Gauss, Sigma_matrix_ih_Gauss, Delta_matrix_ih, 
                                              X_matrix_hh, Sigma_hh_Gauss, Sigma_matrix_hh_Gauss, Delta_matrix_hh)

        
        return d_input, d_hidden_state, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None, None