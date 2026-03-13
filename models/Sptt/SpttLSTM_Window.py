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

# V1版本相比于传统的StittLSTM优化了QR分解的效率,但其本质不变， 以及对一些处理预先分配内存。
# V1版本结果很令人兴奋，结果和StittLSTM一样，但更快了。
# 探究采样行数对准确性，收敛速度，传播时间的影响。

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
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device,config)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")

    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        # 返回的是一个元组，主程序训练的时候，已经实现了解包操作。
        result = self.lstm(x, target, criterion, model_optim, inp=inp, mask=mask)
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
        
        return result
    
class CustomLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, time_block_num, device, config):
        super(CustomLSTM, self).__init__()
        
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
        
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, feature_dim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))
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

    def forward(self, x, targets=None, criterion=None, model_optim=None, inp=None, mask=None):
        # x的shape表示[batch_size, seq_len, feature_dim]
        global_param['time_for_batch'] = 0.0
        
        if self.task_name == 'imputation':
            x_new = x
            x = inp
        
        # 获取输入序列的第一个尺度batch_size，进行隐藏状态和细胞状态的初始化
        num_samples = x.size(0)
        seq_len = x.size(1)
        
        # 在RNN做时间序列预测中，要对每个样本初始化一个隐藏状态和细胞状态
        hx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        cx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        
        # 总的元素数
        # T = num_samples * seq_len # 这个其实就是按行拼接完所有时间步最后所有的行数。
        
        # truncate_length = seq_len // self.time_block_num
        
        # 划分时间块的索引
        indices = np.linspace(0, seq_len, self.time_block_num + 1, dtype=int)
        
        sum_loss = 0.0
        
        # total_chunks = (seq_len + truncate_length - 1) // truncate_length
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
                    cx.detach_()
                
                # 处理当前chunk的时间步
                time_step = 0
                for j in range(chunk_start, chunk_end):
                    # 这是一个三维的张量，表示取x的全部batch，第一个时间步的全部特征维度。
                    # 这样数据的话每次就是
                    # [1 1 1 1
                    # 2 2 2 2
                    # 3 3 3 3]
                    time_step += 1
                    # 只有反向传播到第一个时间步的时候，表示当前时间块处理完毕，再进行梯度更新
                    inputs = x[:, j, :]
                    hx, cx = LSTMCellFunction.apply(
                        inputs, hx, cx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step, chunk_block
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
            time_step = 0
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx, cx = LSTMCellFunction.apply(
                    inputs, hx, cx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step
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
            
            time_step = 1
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx, cx = LSTMCellFunction.apply(
                    inputs, hx, cx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step
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
    
class LSTMCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hx, cx, w_ih, w_hh, b_ih, b_hh, time_step=None, chunk_block=None):
        gates = (torch.mm(inputs, w_ih.t()) + b_ih + torch.mm(hx, w_hh.t()) + b_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        ctx.save_for_backward(inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh)
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block
        
        # 返回一个零时间，真正的时间将在backward中测量。
        return hy, cy
    
    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh = ctx.saved_tensors
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block
        
        # 初始化梯度
        grad_inputs = grad_hx = grad_cx = grad_w_ih = grad_w_hh = grad_b_ih = grad_b_hh = None
        
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
        
        # print("时间步:", time_step) 打印的OK
        
        # 如果时间步等于1表示当前时间块的数据都存储完毕。
        if time_step == 1:
            grad_w_ih = torch.zeros_like(w_ih)
            grad_w_hh = torch.zeros_like(w_hh)
            
            # 取stitt所需参数
            X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih, X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh = accumulator.deliver_sbtpca_parameter()
            # input_cat, hx_cat, delta_cat = accumulator.get_concatenated_gradients()
            activation_input, activation_hx, delta = accumulator.get_concatenated_gradients()
            
            # len是返回张量的第一个维度的长度，对应的就是batch_size行数。
            # T: 表示存储数据的全部行数, t：表示每个块的实际长度, i表示当前块的索引号
            # T = len(input_cat) 
            # t = T // time_block_num 
            # print("T // t", T // t)
            t = len(activation_input)
            i = chunk_block
            # print("块索引: ", chunk_block) 打印的OK
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
                
                X_matrix_ih, _, Sigma_ih_signal_X = QR_matrix(X_matrix_ih, X_matrix_ih.shape)
                Delta_matrix_ih, _, Sigma_ih_signal_Delta = QR_matrix(Delta_matrix_ih, Delta_matrix_ih.shape)
                
                align_ih = Sigma_ih_signal_X * Sigma_ih_signal_Delta
                Delta_matrix_ih = Delta_matrix_ih * align_ih.unsqueeze(0)
                
                X_matrix_hh, _, Sigma_hh_signal_X = QR_matrix(X_matrix_hh, X_matrix_hh.shape)
                Delta_matrix_hh, _, Sigma_hh_signal_Delta = QR_matrix(Delta_matrix_hh, Delta_matrix_hh.shape)
                
                align_hh = Sigma_hh_signal_X * Sigma_hh_signal_Delta
                Delta_matrix_hh = Delta_matrix_hh * align_hh.unsqueeze(0)
                    
                # 批量计算Sigma更新
                Sigma_ih_product = torch.sum(((delta @ Delta_matrix_ih) * (activation_input @ X_matrix_ih)) / ((i + 1) * t), dim=0)
                Sigma_ih = i_factor * Sigma_ih + Sigma_ih_product      
                              
                Sigma_hh_product = torch.sum(((delta @ Delta_matrix_hh) * (activation_hx @ X_matrix_hh)) / ((i + 1) * t), dim=0)
                Sigma_hh = i_factor * Sigma_hh + Sigma_hh_product
                
                # XXX:
                Sigma_ih[Sigma_ih == 0] = 1
                Sigma_hh[Sigma_hh == 0] = 1
                    
                # 在计算完成后、更新梯度前应用对数缩放
                Sigma_matrix_ih = torch.diag(Sigma_ih)
                Sigma_matrix_hh = torch.diag(Sigma_hh)
                    
                # 更新逆矩阵
                inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
                inv_Sigma_matrix_hh = torch.inverse(Sigma_matrix_hh)
            
            
            grad_w_ih = (X_matrix_ih @ Sigma_matrix_ih @ Delta_matrix_ih.t()).t()
            grad_w_hh = (X_matrix_hh @ Sigma_matrix_hh @ Delta_matrix_hh.t()).t()
            
            end_calculate_gradients = time.time()
            time_backward = end_calculate_gradients - start_calculate_gradients
            
            global_param['time_for_batch'] += time_backward
            
            if global_param['viz_spectrum_drift']:
                if 'sigma_list' not in global_param:
                    global_param['sigma_list'] = []
                global_param['sigma_list'].append(Sigma_hh.detach().cpu().numpy())
            
            accumulator.save_sbtpca_parameter(X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih, 
                                              X_matrix_hh, Sigma_hh, Sigma_matrix_hh, Delta_matrix_hh)
            
        return grad_inputs, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None, None