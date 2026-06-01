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

        if self.task_name == 'long_term_forecast':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim        
            self.sru = CustomSRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size, config)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.sru = CustomSRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim
            self.sru = CustomSRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.sru = CustomSRU(self.feature_dim, self.hidden_dim, self.krank, self.device, config)
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")
    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        result = self.sru(x, target, criterion, model_optim, inp=inp, mask=mask)
        return result

class CustomSRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, device, config):
        super(CustomSRU, self).__init__()
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
        
        # transformed input
        self.w_x = nn.Parameter(torch.Tensor(hidden_dim, feature_dim))
        self.b_x = nn.Parameter(torch.Tensor(hidden_dim))
        
        # forget gate
        self.w_f = nn.Parameter(torch.Tensor(hidden_dim, feature_dim))
        self.v_f = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))
        
        # reset gate
        self.w_r = nn.Parameter(torch.Tensor(hidden_dim, feature_dim))
        self.v_r = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_r = nn.Parameter(torch.Tensor(hidden_dim))
        
        # skip connection gate projection if feature_dim != hidden_dim
        if feature_dim != hidden_dim:
            self.w_skip = nn.Parameter(torch.Tensor(hidden_dim, feature_dim))
            self.b_skip = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.w_skip = None
            self.b_skip = None
        
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
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, targets=None, criterion=None, model_optim=None, inp=None, mask=None):
        global_param['time_for_batch'] = 0.0
        
        if self.task_name == 'imputation':
            x_new = x
            x = inp
        
        num_samples = x.size(0)
        seq_len = x.size(1)
        
        hx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        cx = torch.zeros(num_samples, self.hidden_dim).to(self.device)
        
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
                    cx.detach_()
                
                time_step = 0
        
                for i in range(chunk_start, chunk_end):
                    time_step += 1
                    inputs = x[:, i, :] # (batch_size, feature_dim)
                    
                    hx, cx = SRUCellFunction.apply(
                        inputs,
                        cx,
                        self.w_x,
                        self.b_x,
                        self.w_f,
                        self.v_f,
                        self.b_f,
                        self.w_r,
                        self.v_r,
                        self.b_r,
                        self.w_skip,
                        self.b_skip,
                        time_step, chunk_block
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
                hx, cx = SRUCellFunction.apply(
                    inputs,
                    cx,
                    self.w_x,
                    self.b_x,
                    self.w_f,
                    self.v_f,
                    self.b_f,
                    self.w_r,
                    self.v_r,
                    self.b_r,
                    self.w_skip,
                    self.b_skip,
                    time_step
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
                hx, cx = SRUCellFunction.apply(
                    inputs,
                    cx,
                    self.w_x,
                    self.b_x,
                    self.w_f,
                    self.v_f,
                    self.b_f,
                    self.w_r,
                    self.v_r,
                    self.b_r,
                    self.w_skip,
                    self.b_skip,
                    time_step
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
        

class SRUCellFunction(torch.autograd.Function):
    """
    the implementation of SRUCell's forward and backward.
    
    It should be noted that the implementation is a bit different from the original paper.
    
    original paper:
        x_hat = W_x * x + b_x
        f_pre = W_f * x + b_f + v_f * cx
        r_pre = W_r * x + b_r + v_r * cx
        forgetgate = sigmoid(f_pre)
        resetgate = sigmoid(r_pre)
        cy = forgetgate * cx + (1.0 - forgetgate) * x_hat
        hy = resetgate * cy + (1.0 - resetgate) * x
        
    modern implementation:
        x_hat = W_x * x + b_x
        f_pre = W_f * x + b_f + v_f * cx
        r_pre = W_r * x + b_r + v_r * cx
        forgetgate = sigmoid(f_pre)
        resetgate = sigmoid(r_pre)
        cy = forgetgate * cx + (1.0 - forgetgate) * x_hat
        hy = resetgate * tanh(cy) + (1.0 - resetgate) * x_skip
        
    where x_skip is either the input x (if input_dim == hidden_dim) or a linear projection of x (if input_dim != hidden_dim).
    
    """
    @staticmethod
    def forward(
        ctx, 
        inputs, cx, 
        w_x, b_x, 
        w_f, v_f, b_f, 
        w_r, v_r, b_r, 
        w_skip, b_skip,
        time_step=None, chunk_block=None
    ):

        # transformed input
        x_hat = torch.mm(inputs, w_x.t()) + b_x # (batch_size, hidden_dim)
        
        # gates
        f_pre = torch.mm(inputs, w_f.t()) + b_f + cx * v_f
        r_pre = torch.mm(inputs, w_r.t()) + b_r + cx * v_r
        
        forgetgate = torch.sigmoid(f_pre)
        resetgate = torch.sigmoid(r_pre)
        
        # cell state update
        cy = forgetgate * cx + (1.0 - forgetgate) * x_hat
        
        ctx.has_skip_proj = (w_skip is not None)
        
        # skip connection
        if ctx.has_skip_proj:
            x_skip = torch.mm(inputs, w_skip.t()) + b_skip
        else:
            if inputs.size(1) != x_hat.size(1):
                raise ValueError(
                    f"input_dim ({inputs.size(1)}) must equal hidden_dim ({x_hat.size(1)}) when w_skip is None"
                )
            x_skip = inputs
        
        tanh_cy = torch.tanh(cy)
        hy = resetgate * tanh_cy + (1.0 - resetgate) * x_skip
        
        if ctx.has_skip_proj:
            ctx.save_for_backward(
                inputs, cx, 
                x_hat, x_skip, 
                forgetgate, resetgate, 
                cy, tanh_cy, 
                w_x, b_x, 
                w_f, v_f, b_f, 
                w_r, v_r, b_r, 
                w_skip, b_skip
            )
        else:
            ctx.save_for_backward(
                inputs, cx, 
                x_hat, x_skip, 
                forgetgate, resetgate, 
                cy, tanh_cy, 
                w_x, b_x, 
                w_f, v_f, b_f, 
                w_r, v_r, b_r, 
            )
        
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block
        
        return hy, cy
    
    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        has_skip_proj = ctx.has_skip_proj
        if has_skip_proj:
            (inputs, cx, 
             x_hat, x_skip, 
             forgetgate, resetgate, 
             cy, tanh_cy, 
             w_x, b_x, 
             w_f, v_f, b_f, 
             w_r, v_r, b_r, 
             w_skip, b_skip) = ctx.saved_tensors
        else:
            (inputs, cx, 
             x_hat, x_skip, 
             forgetgate, resetgate, 
             cy, tanh_cy, 
             w_x, b_x, 
             w_f, v_f, b_f, 
             w_r, v_r, b_r) = ctx.saved_tensors
            
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block
        
        # h = resetgate * tanh(cy) + (1.0 - resetgate) * x_skip
        # dL/dr = dL/dh * dh/dr
        grad_resetgate = grad_hy * (tanh_cy - x_skip)
        # dL/dtanh_cy = dL/dh * dh/dtanh_cy
        grad_tanh_cy = grad_hy * resetgate
        grad_x_skip = grad_hy * (1.0 - resetgate)
        
        # tanh(x)
        grad_cy_total = grad_tanh_cy * (1.0 - tanh_cy ** 2) + grad_cy
        
        # c = f * cx + (1-f) * x_hat
        grad_forgetgate = grad_cy_total * (cx - x_hat)
        grad_cx = grad_cy_total * forgetgate
        grad_x_hat = grad_cy_total * (1.0 - forgetgate)

        # sigmoid
        grad_f_pre = grad_forgetgate * forgetgate * (1.0 - forgetgate)
        grad_r_pre = grad_resetgate * resetgate * (1.0 - resetgate)

        # recurrent vector part from cx * v
        grad_cx = grad_cx + grad_f_pre * v_f + grad_r_pre * v_r
        grad_v_f = torch.sum(grad_f_pre * cx, dim=0)
        grad_v_r = torch.sum(grad_r_pre * cx, dim=0)
        
        # weight and bias gradients
        start_calculate_gradients = time.time()
        
        grad_w_x = torch.mm(grad_x_hat.t(), inputs)
        grad_b_x = grad_x_hat.sum(0)

        grad_w_f = torch.mm(grad_f_pre.t(), inputs)
        grad_b_f = grad_f_pre.sum(0)

        grad_w_r = torch.mm(grad_r_pre.t(), inputs)
        grad_b_r = grad_r_pre.sum(0)
        
        if has_skip_proj:
            grad_w_skip = torch.mm(grad_x_skip.t(), inputs)
            grad_b_skip = grad_x_skip.sum(0)
        else:
            grad_w_skip = None
            grad_b_skip = None
            
        end_calculate_gradients = time.time()
        global_param['time_for_batch'] = end_calculate_gradients - start_calculate_gradients
        
        # input gradients
        grad_inputs = (
            torch.mm(grad_x_hat, w_x) +
            torch.mm(grad_f_pre, w_f) +
            torch.mm(grad_r_pre, w_r)
        )
        
        if has_skip_proj:
            grad_inputs = grad_inputs + torch.mm(grad_x_skip, w_skip)
        else:
            grad_inputs = grad_inputs + grad_x_skip

        return (
            grad_inputs,   # inputs
            grad_cx,       # cx
            grad_w_x,      # w_x
            grad_b_x,      # b_x
            grad_w_f,      # w_f
            grad_v_f,      # v_f
            grad_b_f,      # b_f
            grad_w_r,      # w_r
            grad_v_r,      # v_r
            grad_b_r,      # b_r
            grad_w_skip,   # w_skip
            grad_b_skip,   # b_skip
            None, None
        )
        