import torch
import torch.nn as nn
import math
from global_param import global_param
import time
from utils.GradientAccumulator_MinGRU_Hybrid import GradientAccumulator_MinGRU_Hybrid as GradientAccumulator
import numpy as np

accumulator = GradientAccumulator()

def chodral_subspace_drift(Q_matrix_old: torch.Tensor, Q_matrix_new: torch.Tensor) -> torch.Tensor:
    """
    [unit_dim, krank]
    Q_matrix_old, Q_matrix_new: [d, k] with orthonormal columns.
    Returns scalar drift (chordal distance): sqrt(k - ||Q_old^T Q_new||_F^2)
    M: Q_old.T @ Q_new
    """
    k = Q_matrix_old.shape[1]
    M = Q_matrix_old.transpose(0, 1) @ Q_matrix_new
    frob_sq = torch.sum(M ** 2)
    drift_sq = torch.clamp(k - frob_sq, min=0.0)
    return torch.sqrt(drift_sq)


"""Version 1"""
def QR_matrix(X_matrix, target_shape):
    Q, R, diagonal_sign = qr_with_non_negative_diagonal(X_matrix)
    # ic(Q.shape, target_shape)
    
    # 使用预分配和高效索引
    if Q.shape != target_shape:
        # 进入这个会出现nan数值错误
        assert Q.shape == target_shape, "not permit to change matrix shape"
        result = torch.zeros(target_shape, device=X_matrix.device, dtype=X_matrix.dtype)
        result[:min(Q.shape[0], target_shape[0]), :min(Q.shape[1], target_shape[1])] = \
            Q[:min(Q.shape[0], target_shape[0]), :min(Q.shape[1], target_shape[1])]
        return result, R, diagonal_sign
    # ic(Q.shape, diagonal_sign.shape)
    return Q, R, diagonal_sign
    
def qr_with_non_negative_diagonal(matrix):
    # 进行QR分解
    Q, R = torch.linalg.qr(matrix)
    
    # 获取 R 矩阵对角线元素的符号
    diagonal_sign = torch.sign(torch.diag(R))
    
    # 避免对角线元素为零的情况
    diagonal_sign[diagonal_sign == 0] = 1
    # ic(diagonal_sign)
    
    # 调整 Q 矩阵的列符号和Q = Q * diagonal_sign得到的结果一样。
    Q1 = Q * diagonal_sign.unsqueeze(0)
    
    R = diagonal_sign.unsqueeze(1) * R
    
    return Q1, R, diagonal_sign


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
            self.gru = CustomMinGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            # 类别数量
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.gru = CustomMinGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.gru = CustomMinGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.gru = CustomMinGRU(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")

    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        # final_hidden = self.gru(x)
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
        # print ("y shape: ", y.shape)
        # y shape [64, 96, 7]
        # return y
        
        result = self.gru(x, target, criterion, model_optim, inp=inp, mask=mask)
        
        return result
    
    
class CustomMinGRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, time_block_num, device, config):
        super(CustomMinGRU, self).__init__()
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
        # minGRU only has input-to-hidden parameters:
        # z_t = sigmoid(W_z x_t + b_z)
        # h_tilde = W_h x_t + b_h
        self.w_ih = nn.Parameter(torch.Tensor(2 * hidden_dim, feature_dim))
        self.b_ih = nn.Parameter(torch.Tensor(2 * hidden_dim))
        
        if self.task_name == 'classification':
            self.fc = nn.Linear(self.hidden_dim, self.config.num_classes)
        elif self.task_name == 'long_term_forecast':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size_anomaly)
        elif self.task_name == 'imputation':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size_anomaly)
        
        self.reset_parameters()
        self.init_sptt_parameters()
        
        accumulator.save_sptt_parameter(
            self.X_matrix_ih, self.Sigma_ih, self.Sigma_matrix_ih, self.Delta_matrix_ih
        )
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def init_sptt_parameters(self):
        self.X_matrix_ih = torch.randn(self.feature_dim, self.krank).to(self.device)
        self.Sigma_ih = torch.randn(self.krank).to(self.device)
        self.Sigma_matrix_ih = torch.diag(self.Sigma_ih).to(self.device)
        self.Delta_matrix_ih = torch.randn(2 * self.hidden_dim, self.krank).to(self.device)
        
    def forward(self, x, targets=None, criterion=None, model_optim=None, inp=None, mask=None):
        # accumulator.reset()
        # global_param['flag'] = True
        # global_param['compression_finished'] = False
        
        global_param['time_for_batch'] = 0.0
        
        if self.task_name == 'imputation':
            x_new = x
            x = inp

        num_samples = x.size(0)
        seq_len = x.size(1)
        
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
                    time_step += 1
                    inputs = x[:, i, :]
                    hx = MinGRUCellFunction.apply(
                        inputs, hx, self.w_ih, self.b_ih, time_step, chunk_block
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
                
                # print("loss is tensor?", type(loss))
                # print("sum_loss is tensor?", type(sum_loss))

                
                if not is_last_chunk:
                    model_optim.zero_grad()
                    # 如果不是最后一个chunk进行反向传播更新梯度和参数
                    loss.backward()
                    model_optim.step()
                    
            # 如果是最后一个last_chunk，直接返回loss，在主程序反向传播，更新梯度和权重 
            if self.task_name == 'classification':
                return outputs, loss, sum_loss
            else:
                return loss, sum_loss
            
        elif model_optim is None and criterion is not None:
            # 进行验证
            # accumulator.reset()
            time_step = 0
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx = MinGRUCellFunction.apply(
                    inputs, hx, self.w_ih, self.b_ih, time_step
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
            
            time_step = 0
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx = MinGRUCellFunction.apply(
                    inputs, hx, self.w_ih, self.b_ih, time_step
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

class MinGRUCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hidden_state, w_ih, b_ih, time_step=None, chunk_block=None):
        gates = torch.mm(inputs, w_ih.t()) + b_ih
        update_gate_pre, h_tilde = gates.chunk(2, 1)
        
        update_gate = torch.sigmoid(update_gate_pre)
        
        # minGRU recurrence
        hy = (1 - update_gate) * hidden_state + update_gate * h_tilde
        
        ctx.save_for_backward(inputs, hidden_state, update_gate, h_tilde, w_ih)
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block
        
        return hy
    
    @staticmethod
    def backward(ctx, grad_hy):
        inputs, hidden_state, update_gate, h_tilde, w_ih = ctx.saved_tensors
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block
        
        d_input = d_hidden_state = grad_w_ih = grad_b_ih = None
        
        # hy = (1 - z) * h_prev + z * h_tilde
        # d hy / d z = h_tilde - h_prev
        d_update_gate = grad_hy * (h_tilde - hidden_state)
        d_update_gate_pre = d_update_gate * update_gate * (1 - update_gate)
        
        # h_tilde is linear
        d_h_tilde = grad_hy * update_gate
        
        # SPTT only needs activation and error
        delta_ih = torch.cat((d_update_gate_pre, d_h_tilde), dim=1)
        accumulator.accumulate(inputs, delta_ih)
        
        d_input = torch.mm(delta_ih, w_ih)
        d_hidden_state = grad_hy * (1 - update_gate)
        grad_b_ih = delta_ih.sum(0)
        
        if time_step == 1:
            grad_w_ih = torch.zeros_like(w_ih)
            
            X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih = accumulator.deliver_sptt_parameter()
            
            inputs_cat, delta_ih_cat = accumulator.get_concatenated_gradients()
            
            i = chunk_block
            
            inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
            
            start_calculate_gradients = time.time()
            
            with torch.no_grad():
                activation_input = inputs_cat
                delta_ih = delta_ih_cat
                
                current_t = activation_input.size(0)
                
                if current_t == 0:
                    raise ValueError("Current time block has zero samples, which should not happen.")
                
                # SPTT-style streaming low-rank update
                scale_factor_right_ih = delta_ih @ Delta_matrix_ih / current_t
                scale_factor_left_ih = activation_input @ X_matrix_ih / current_t
                
                i_factor = i / (i + 1)
                update_factor = 1 / (i + 1)
                
                X_ih_update = activation_input.t() @ scale_factor_right_ih @ inv_Sigma_matrix_ih
                X_matrix_ih = i_factor * X_matrix_ih + update_factor * X_ih_update
                
                Delta_ih_update = delta_ih.t() @ scale_factor_left_ih @ inv_Sigma_matrix_ih
                Delta_matrix_ih = i_factor * Delta_matrix_ih + update_factor * Delta_ih_update
                
                # QR orthogonalization only
                X_matrix_ih, _, Sigma_ih_signal_X = QR_matrix(X_matrix_ih, X_matrix_ih.shape)
                Delta_matrix_ih, _, Sigma_ih_signal_Delta = QR_matrix(Delta_matrix_ih, Delta_matrix_ih.shape)

                align_ih = Sigma_ih_signal_X * Sigma_ih_signal_Delta
                Delta_matrix_ih = Delta_matrix_ih * align_ih.unsqueeze(0)   
                
                # update Sigma
                Sigma_ih_product = torch.sum(
                    ((delta_ih @ Delta_matrix_ih) * (activation_input @ X_matrix_ih)) / ((i + 1) * current_t),
                    dim=0
                )
                Sigma_ih = i_factor * Sigma_ih + Sigma_ih_product            
                
                Sigma_ih[Sigma_ih == 0] = 1
                
                Sigma_matrix_ih = torch.diag(Sigma_ih)
                
                inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
                    
            grad_w_ih = (X_matrix_ih @ Sigma_matrix_ih @ Delta_matrix_ih.t()).t()
            
            end_calculate_gradients = time.time()
            
            global_param['time_for_batch'] = end_calculate_gradients - start_calculate_gradients
            
            if global_param.get('viz_spectrum_drift', False):
                if 'sigma_list' not in global_param:
                    global_param['sigma_list'] = []
                global_param['sigma_list'].append(Sigma_ih.detach().cpu().numpy())

            accumulator.save_sptt_parameter(
                X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih
            )
            
        return d_input, d_hidden_state, grad_w_ih, grad_b_ih, None, None                              