import os
import torch
import sys
import torch.nn as nn
import math
from global_param import global_param
import time
import numpy as np
from icecream import ic

# class Model(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=None):
#         super(Model, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         pass

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.seq_len = config.seq_len
        print("seq_len: ", self.seq_len)
        self.hidden_dim = config.hidden_dim
        self.device = config.gpu_type
        self.krank = config.krank
        self.task_name = config.task_name
        # 如果是多变量预测多变量的问题，全连接层的输出尺寸应该是 特征维度 X 预测长度
        if self.task_name == 'long_term_forecast':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim        
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
        
            # 这个地方对于多变量预测单变量完全可以，如果这个地方不修改的话，需要修改train程序的输出取值
            # self.fc = nn.Linear(self.hidden_dim, self.output_size)
            
            # 为了train程序对应，做以下修改
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            # print("进入anomaly_detection")
            self.output_size = config.seq_len
            # print("output_size: ", self.output_size)
            self.feature_dim = config.feature_dim
            # print("feature_dim: ", self.feature_dim)
            self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")


    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        loss = self.lstm(x, target, criterion, model_optim, inp=inp, mask=mask)

        # print("fc_out shape: ", fc_out.shape)
        # fc_out shape [64, 672]

        # print ("y shape: ", y.shape)
        # y shape [64, 96, 7]
        return loss
        
class CustomLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, device, config):
        super(CustomLSTM, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.krank = krank
        self.time_block_num = config.time_block_num
        self.device = device
        
        self.task_name = config.task_name
        self.output_size = config.pred_len
        self.feature_dim = config.feature_dim
        self.output_size_anomaly = config.seq_len
        self.num_classes = config.num_classes
        self.config = config
        
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, feature_dim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))
        
        if self.task_name == 'long_term_forecast':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        elif self.task_name == 'imputation':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size_anomaly)
        elif self.task_name == 'anomaly_detection':
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size_anomaly)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # 定义标准差
        std = 1.0 / math.sqrt(self.hidden_dim)
        # 初始化参数
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, targets=None, criterion=None, model_optim=None, inp=None, mask=None):
        # print("x shape: ", x.shape)
        # x的shape应该是(batch_size, seq_len, feature_dim) (64, 384, 4)
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
        
        # 划分时间块的索引
        indices = np.linspace(0, seq_len, self.time_block_num + 1, dtype=int)
        
        sum_loss = 0.0        
        if model_optim is not None:
            chunk_block = 0
            
            for i in range(self.time_block_num):
                chunk_start = indices[i]
                chunk_end = indices[i + 1]
                chunk_block += 1
                
                # 判断是否是最后一个chunk
                if i == self.time_block_num - 1:
                    is_last_chunk = True
                else:
                    is_last_chunk = False
                    
                # 截断梯度流，但保持状态连续性
                if chunk_start > 0:
                    # 保留数值前向继续，但断开反向传播图，这样确保每个时间块之间状态连续但梯度不连续。
                    hx.detach_()
                    cx.detach_()
                
                # 处理当前chunk时间步
                time_step = 0
                for j in range(chunk_start, chunk_end):
                    time_step += 1
                    inputs = x[:, j, :]
                    
                    hx, cx = LSTMCellFunction.apply(
                        inputs, hx, cx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step, chunk_block
                    )
                    
                fc_out = self.fc(hx)
                
                # 计算损失(训练)
                if self.task_name == "long_term_forecast":
                    y = fc_out.view(-1, self.output_size, self.feature_dim)
                    f_dim = -1 if self.config.features == 'MS' else 0
                    outputs = y[:, -self.output_size:, f_dim:]
                    targets = targets[:, -self.output_size:, f_dim:]
                    loss = criterion(outputs, targets)
                elif self.task_name == "classification":
                    outputs = fc_out
                    # 将targets张量的数据类型转换为​​长整型（long）​​（即int64）, 通常用于表示标签类别。
                    # 对转换后的张量进行维度压缩，移除最后一个维度中大小为1的维度。（如果存在）
                    # 如形状为[batch_size, 1]，则会被移除，结果形状变为[batch_size]
                    loss = criterion(outputs, targets.long().squeeze(-1))
                elif self.task_name == "anomaly_detection":
                    y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
                    f_dim = -1 if self.config.features == 'MS' else 0
                    outputs = y[:, :, f_dim:]
                    loss = criterion(outputs, x)
                elif self.task_name == "imputation":
                    y = fc_out.view(-1, self.output_size_anomaly, self.feature_dim)
                    f_dim = -1 if self.config.features == 'MS' else 0
                    outputs = y[:, :, f_dim:]
                    batch_x = x_new[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]
                    loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                
                sum_loss += loss.item()
                
                if not is_last_chunk:
                    model_optim.zero_grad()
                    loss.backward()
                    
                    if self.config.viz_spectrum_drift:
                        grad_w_hh = self.w_hh.grad.clone().detach()
                        assert grad_w_hh is not None, "grad_w_hh is None"
                        if "sigma_list" not in global_param:
                            ic("model name: ", self.config.model, "allow to record spectrum drift")
                            global_param["sigma_list"] = []
                        _, Sigma, _ = torch.linalg.svd(grad_w_hh, full_matrices=False)
                        truncated_sigma = Sigma[:self.config.krank]
                        global_param["sigma_list"].append(truncated_sigma.cpu().numpy())
                        
                    model_optim.step()
            
            # 如果是最后一个last_chunk, 直接返回loss, 在主程序反向传播，更新梯度和权重
            if self.task_name == 'classification':
                return outputs, loss, sum_loss
            else:
                return loss, sum_loss
            
        elif model_optim is None and criterion is not None:
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
        
        # 计算各门的梯度
        grad_outgate = grad_hy * torch.tanh(cy) * outgate * (1 - outgate)
        grad_cy = grad_hy * outgate * (1 - torch.tanh(cy) ** 2) + grad_cy
        grad_ingate = grad_cy * cellgate * ingate * (1 - ingate)
        grad_cellgate = grad_cy * ingate * (1 - cellgate ** 2)
        grad_forgetgate = grad_cy * cx * forgetgate * (1 - forgetgate)
        
        # 计算权重的梯度
        start_calculate_gradients = time.time()

        grad_w_ih = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), inputs)
        grad_w_hh = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), hx)
        
        end_calculate_gradients = time.time()
        time_backward = end_calculate_gradients - start_calculate_gradients
        
        # 完成一个batch的所有时间步的梯度计算所需的时间
        global_param['time_for_batch'] += time_backward
        
        # 计算偏置的梯度
        grad_b_ih = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
        grad_b_hh = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
        
        # 计算输入和隐藏状态的梯度
        grad_inputs = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_ih)
        grad_hx = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_hh)
        
        # 计算上一个单元状态的梯度
        grad_cx = grad_cy * forgetgate

        # 返回与forward方法相同的数量的参数
        return grad_inputs, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None, None