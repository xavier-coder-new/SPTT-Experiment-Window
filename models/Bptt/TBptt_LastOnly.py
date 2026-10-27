import torch
import torch.nn as nn
import math
from global_param import global_param
import time
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
        self.feature_dim = config.feature_dim
        self.lstm = CustomLSTM(self.feature_dim, self.hidden_dim, self.krank, self.device, config)

    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        loss = self.lstm(x, target, criterion, model_optim, inp=inp, mask=mask)
        return loss
        
class CustomLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, device, config):
        super(CustomLSTM, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.krank = krank
        self.time_block_num = config.time_block_num
        self.device = device
        self.use_chunk = bool(getattr(config, "use_chunk", 1))
        
        self.task_name = config.task_name
        self.feature_dim = config.feature_dim
        self.num_classes = config.num_classes
        self.config = config
        
        self.w_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, feature_dim))
        self.w_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))
        
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, x, targets=None, criterion=None,
                model_optim=None, inp=None, mask=None):
        global_param['time_for_batch'] = 0.0

        batch_size, seq_len, _ = x.shape

        hx = x.new_zeros(batch_size, self.hidden_dim)
        cx = x.new_zeros(batch_size, self.hidden_dim)

        is_training_forward = model_optim is not None

        if is_training_forward and self.use_chunk:
            indices = np.linspace(
                0, seq_len,
                self.time_block_num + 1,
                dtype=int
            )
            
            final_loss = None
            outputs = None
            total_loss_value = 0.0
            
            # TBPTT
            for chunk_idx in range(self.time_block_num):
                chunk_start = indices[chunk_idx]
                chunk_end = indices[chunk_idx + 1]
                
                if chunk_idx == self.time_block_num - 1:
                    is_last_chunk = True
                else:
                    is_last_chunk = False
                
                if chunk_start > 0:
                    hx = hx.detach()
                    cx = cx.detach()

                for j in range(chunk_start, chunk_end):
                    hx, cx = LSTMCellFunction.apply(
                        x[:, j, :],
                        hx, cx,
                        self.w_ih, self.w_hh,
                        self.b_ih, self.b_hh,
                        j + 1, chunk_idx + 1
                    )
                    
                outputs = self.fc(hx)
                chunk_loss = criterion(outputs, targets.long().squeeze(-1))

                total_loss_value += (
                    chunk_loss.detach().item() / self.time_block_num
                )

                if not is_last_chunk:
                    chunk_loss.backward()
                    model_optim.zero_grad(set_to_none=True)
                else:
                    final_loss = chunk_loss

            return outputs, final_loss, total_loss_value
                
        else:
            for j in range(seq_len):
                hx, cx = LSTMCellFunction.apply(
                    x[:, j, :],
                    hx, cx,
                    self.w_ih, self.w_hh,
                    self.b_ih, self.b_hh,
                    j + 1, None
                )

            outputs = self.fc(hx)

            return outputs


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
        
        return hy, cy
    
    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        
        inputs, hx, cx, hy, cy, ingate, forgetgate, cellgate, outgate, w_ih, w_hh, b_ih, b_hh = ctx.saved_tensors
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block
        
        grad_outgate = grad_hy * torch.tanh(cy) * outgate * (1 - outgate)
        grad_cy = grad_hy * outgate * (1 - torch.tanh(cy) ** 2) + grad_cy
        grad_ingate = grad_cy * cellgate * ingate * (1 - ingate)
        grad_cellgate = grad_cy * ingate * (1 - cellgate ** 2)
        grad_forgetgate = grad_cy * cx * forgetgate * (1 - forgetgate)
        
        start_calculate_gradients = time.time()

        grad_w_ih = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), inputs)
        grad_w_hh = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).t(), hx)
        
        end_calculate_gradients = time.time()
        time_backward = end_calculate_gradients - start_calculate_gradients
        
        global_param['time_for_batch'] += time_backward
        
        grad_b_ih = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
        grad_b_hh = torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1).sum(0)
        
        grad_inputs = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_ih)
        grad_hx = torch.mm(torch.cat((grad_ingate, grad_forgetgate, grad_cellgate, grad_outgate), 1), w_hh)
        
        grad_cx = grad_cy * forgetgate

        return grad_inputs, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh, None, None