import os
import torch
import sys
import torch.nn as nn
import math
from global_param import global_param
import time
import numpy as np
from icecream import ic

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.seq_len = config.seq_len
        print("seq_len: ", self.seq_len)
        self.hidden_dim = config.hidden_dim
        self.device = config.gpu_type
        self.krank = config.krank
        self.task_name = config.task_name
        if self.task_name == 'long_term_forecast':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.gru = CustomGRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")
            
    
    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        loss = self.gru(x, target, criterion, model_optim, inp=inp, mask=mask)
        return loss

class CustomGRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, device, config):
        super(CustomGRU, self).__init__()
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
        
        self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, feature_dim))
        self.w_hh = nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(3 * hidden_dim))
        
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
        global_param['time_for_batch'] = 0.0
        
        if self.task_name == 'imputation':
            x_new = x
            x = inp
        
        num_samples = x.size(0)
        seq_len = x.size(1)
        
        hx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        
        indices = np.linspace(0, seq_len, self.time_block_num + 1, dtype=int)
        
        sum_loss = 0.0  
        
        if model_optim is not None:
            chunk_block = 0
            
            for i in range(self.time_block_num):
                chunk_start = indices[i]
                chunk_end = indices[i + 1]
                chunk_block += 1
                
                if i == self.time_block_num - 1:
                    is_last_chunk = True
                else:
                    is_last_chunk = False
                    
                if chunk_start > 0:
                    hx.detach_()

                time_step = 0
                
                for j in range(chunk_start, chunk_end):
                    time_step += 1
                    inputs = x[:, j, :]
                    
                    hx = GRUCellFunction.apply(
                        inputs, hx, self.w_ih, self.w_hh, self.b_ih, self.b_hh, time_step, chunk_block
                    )
            
                fc_out = self.fc(hx)
            
                if self.task_name == "long_term_forecast":
                    y = fc_out.view(-1, self.output_size, self.feature_dim)
                    f_dim = -1 if self.config.features == 'MS' else 0
                    outputs = y[:, -self.output_size:, f_dim:]
                    targets = targets[:, -self.output_size:, f_dim:]
                    loss = criterion(outputs, targets)
                elif self.task_name == "classification":
                    outputs = fc_out
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
                    model_optim.step()
            
            if self.task_name == 'classification':
                return outputs, loss, sum_loss
            else:
                return loss, sum_loss

        elif model_optim is None and criterion is not None:
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
        
class GRUCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hidden_state, w_ih, w_hh, b_ih, b_hh, time_step=None, chunk_block=None):
        w_ir, w_iz, w_in = w_ih.chunk(3, 0)
        w_hr, w_hz, w_hn = w_hh.chunk(3, 0)
        b_ir, b_iz, b_in = b_ih.chunk(3)
        b_hr, b_hz, b_hn = b_hh.chunk(3)
        
        reset_gate = torch.sigmoid(
            torch.mm(input, w_ir.t()) + b_ir + torch.mm(hidden_state, w_hr.t()) + b_hr
        )
        
        update_gate = torch.sigmoid(
            torch.mm(input, w_iz.t()) + b_iz + torch.mm(hidden_state, w_hz.t()) + b_hz
        )
        
        n = torch.tanh(
            torch.mm(input, w_in.t()) + b_in + reset_gate * (torch.mm(hidden_state, w_hn.t()) + b_hn)
        )
        
        hy = (1 - update_gate) * n + update_gate * hidden_state

        ctx.save_for_backward(input, hidden_state, w_ir, w_iz, w_in, w_hr, w_hz, w_hn,
                              reset_gate, update_gate, n, b_ir, b_iz, b_in, b_hr, b_hz, b_hn)
        
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block
        
        return hy

    @staticmethod
    def backward(ctx, grad_hy):
        input, hidden_state, w_ir, w_iz, w_in, w_hr, w_hz, w_hn, \
        reset_gate, update_gate, n, b_ir, b_iz, b_in, b_hr, b_hz, b_hn = ctx.saved_tensors
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block

        d_input = d_hidden_state = grad_w_ih = grad_w_hh = grad_b_ih = grad_b_hh = None

        d_update_gate = (hidden_state - n) * grad_hy
        d_update_gate = d_update_gate * update_gate * (1 - update_gate)

        d_n = (1 - update_gate) * grad_hy
        d_n_tanh = d_n * (1 - n ** 2)

        hidden_w_hn = torch.mm(hidden_state, w_hn.t()) + b_hn
        d_reset_gate = d_n_tanh * hidden_w_hn
        d_reset_gate = d_reset_gate * reset_gate * (1 - reset_gate)

        d_input = (
            torch.mm(d_reset_gate, w_ir) +
            torch.mm(d_update_gate, w_iz) +
            torch.mm(d_n_tanh, w_in)
        )

        d_hidden_state = (
            grad_hy * update_gate +
            torch.mm(d_reset_gate, w_hr) +
            torch.mm(d_update_gate, w_hz) +
            reset_gate * torch.mm(d_n_tanh, w_hn)
        )

        start_calculate_gradients = time.time()
        
        grad_w_ir = torch.mm(d_reset_gate.t(), input)
        grad_w_iz = torch.mm(d_update_gate.t(), input)
        grad_w_in = torch.mm(d_n_tanh.t(), input)
        grad_w_ih = torch.cat((grad_w_ir, grad_w_iz, grad_w_in), dim=0)

        grad_w_hr = torch.mm(d_reset_gate.t(), hidden_state)
        grad_w_hz = torch.mm(d_update_gate.t(), hidden_state)
        grad_w_hn = torch.mm((d_n_tanh * reset_gate).t(), hidden_state)
        grad_w_hh = torch.cat((grad_w_hr, grad_w_hz, grad_w_hn), dim=0)

        end_calculate_gradients = time.time()
        time_backward = end_calculate_gradients - start_calculate_gradients
        
        global_param['time_for_batch'] += time_backward
        
        grad_b_ir = d_reset_gate.sum(0)
        grad_b_iz = d_update_gate.sum(0)
        grad_b_in = d_n_tanh.sum(0)
        grad_b_ih = torch.cat((grad_b_ir, grad_b_iz, grad_b_in))

        grad_b_hr = d_reset_gate.sum(0)
        grad_b_hz = d_update_gate.sum(0)
        grad_b_hn = (d_n_tanh * reset_gate).sum(0)
        grad_b_hh = torch.cat((grad_b_hr, grad_b_hz, grad_b_hn))

        return d_input, d_hidden_state, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None, None
