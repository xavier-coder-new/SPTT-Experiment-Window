import torch
import torch.nn as nn
import math
from global_param import global_param
import time
from icecream import ic
import numpy as np

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
            self.lstm = CustomMinLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
        
            # 这个地方对于多变量预测单变量完全可以，如果这个地方不修改的话，需要修改train程序的输出取值
            # self.fc = nn.Linear(self.hidden_dim, self.output_size)
            
            # 为了train程序对应，做以下修改
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.lstm = CustomMinLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomMinLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            # print("进入anomaly_detection")
            self.output_size = config.seq_len
            # print("output_size: ", self.output_size)
            self.feature_dim = config.feature_dim
            # print("feature_dim: ", self.feature_dim)
            self.lstm = CustomMinLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")


    # def forward(self, x, epoch_count=None, batch_idx=None):
    #     final_hidden = self.lstm(x)
    #     fc_out = self.fc(final_hidden)

    #     if self.task_name == 'long_term_forecast':
    #         y = fc_out.view(-1, self.output_size, self.feature_dim)
    #     elif self.task_name == 'classification':
    #         y = fc_out
    #     elif self.task_name == 'imputation':
    #         y = fc_out.view(-1, self.output_size, self.feature_dim)
    #     elif self.task_name == 'anomaly_detection':
    #         y = fc_out.view(-1, self.output_size, self.feature_dim)

    #     return y
    
    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        loss = self.lstm(x, target, criterion, model_optim, inp=inp, mask=mask)

        # print("fc_out shape: ", fc_out.shape)
        # fc_out shape [64, 672]

        # print ("y shape: ", y.shape)
        # y shape [64, 96, 7]
        return loss
    

class CustomMinLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, device, config):
        super(CustomMinLSTM, self).__init__()
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

        # minLSTM only depends on x_t, not on h_{t-1}, so we only need three weights.
        # forget_weight, input_weight, candidate_weight
        self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, feature_dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
        
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
        std = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
            
    def forward(self, x, targets=None, criterion=None, model_optim=None, inp=None, mask=None):
        # x shape: (batch_size, seq_len, feature_dim)
        global_param['time_for_batch'] = 0.0
        
        if self.task_name == 'imputation':
            x_new = x
            x = inp
        
        num_samples = x.size(0)
        seq_len = x.size(1)
        
        # minLSTM only keeps hidden states, and does not need cell states.
        hx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        
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
                
                # 处理当前chunk时间步
                time_step = 0
                for i in range(chunk_start, chunk_end):
                    time_step += 1
                    inputs = x[:, i, :]
                    hx = MinLSTMCellFunction.apply(
                        inputs, hx, self.w_ih, self.b_ih, time_step, chunk_block
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
                        grad_w_ih = self.w_ih.grad.clone().detach()
                        assert grad_w_ih is not None, "grad_w_ih is None"
                        if "sigma_list" not in global_param:
                            ic("model name: ", self.config.model, "allow to record spectrum drift")
                            global_param["sigma_list"] = []
                        _, Sigma, _ = torch.linalg.svd(grad_w_ih, full_matrices=False)
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
                hx = MinLSTMCellFunction.apply(
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
            time_step = 0
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx = MinLSTMCellFunction.apply(
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

class MinLSTMCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hx, w_ih, b_ih, time_step=None, chunk_block=None):
        gates = torch.mm(inputs, w_ih.t()) + b_ih
        forgetgate_pre, inputgate_pre, candidategate = gates.chunk(3, 1)
        
        forgetgate = torch.sigmoid(forgetgate_pre)
        inputgate = torch.sigmoid(inputgate_pre)
        
        h_tilde = candidategate
        
        # Normalize to ensure that f' + i' = 1
        denom = forgetgate + inputgate + 1e-8
        f_prime = forgetgate / denom
        i_prime = inputgate / denom
        
        hy = f_prime * hx + i_prime * h_tilde
        
        ctx.save_for_backward(
            inputs, hx, w_ih, b_ih,
            forgetgate_pre, inputgate_pre,
            forgetgate, inputgate,
            h_tilde, f_prime, i_prime, denom
        )
        
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block

        return hy
    
    @staticmethod
    def backward(ctx, grad_hy):
        (
            inputs, hx, w_ih, b_ih,
            forgetgate_pre, inputgate_pre,
            forgetgate, inputgate,
            h_tilde, f_prime, i_prime, denom
        ) = ctx.saved_tensors
        
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block
        
        # hy = f' * hx + i' * h_tilde
        grad_hx = grad_hy * f_prime
        grad_h_tilde = grad_hy * i_prime
        
        # the graidents of f' and i'
        grad_f_prime = grad_hy * hx
        grad_i_prime = grad_hy * h_tilde
        
        # f' = f / (f + i), i' = i / (f + i)
        # Set s = f + i
        # dL/df = dL/df' * df'/df + dL/di' * di'/df
        # dL/df = (i / s^2) * (dL/df' - dL/di')
        # dL/di = (f / s^2) * (dL/di' - dL/df')
        
        s2 = denom * denom
        
        grad_forgetgate = (inputgate / s2) * (grad_f_prime - grad_i_prime)
        grad_inputgate = (forgetgate / s2) * (grad_i_prime - grad_f_prime)
        
        # sigmoid reverse transmission
        grad_forgetgate_pre = grad_forgetgate * forgetgate * (1 - forgetgate)
        grad_inputgate_pre = grad_inputgate * inputgate * (1 - inputgate)
        
        # The candidate is linear.
        grad_candidategate = grad_h_tilde
        
        grad_gates = torch.cat(
            (grad_forgetgate_pre, grad_inputgate_pre, grad_candidategate), dim=1
        )
        
        start_calculate_gradients = time.time()
        
        grad_w_ih = torch.mm(grad_gates.t(), inputs)
        
        end_calculate_gradients = time.time()
        
        time_backward = end_calculate_gradients - start_calculate_gradients
        
        global_param['time_for_batch'] += time_backward
        
        grad_b_ih = grad_gates.sum(0)
        
        grad_inputs = torch.mm(grad_gates, w_ih)
        
        return grad_inputs, grad_hx, grad_w_ih, grad_b_ih, None, None