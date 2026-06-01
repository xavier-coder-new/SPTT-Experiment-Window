import torch
import torch.nn as nn
import math
from global_param import global_param
import time
from utils.GradientAccumulator_SRU import GradientAccumulatorSRU_Hybrid as GradientAccumulatorSRU
from icecream import ic
import numpy as np

accumulator = GradientAccumulatorSRU()

def QR_matrix(X_matrix, target_shape):
    Q, R, diagonal_sign = qr_with_non_negative_diagonal(X_matrix)
    
    return Q, R, diagonal_sign
    
def qr_with_non_negative_diagonal(matrix):
    Q, R = torch.linalg.qr(matrix)
    
    diagonal_sign = torch.sign(torch.diag(R))
    
    diagonal_sign[diagonal_sign == 0] = 1

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

        if self.task_name == 'long_term_forecast':
            self.output_size = config.pred_len
            self.feature_dim = config.feature_dim
            self.sru = CustomSRU(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.sru = CustomSRU(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.output_size)
        elif self.task_name == 'imputation':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.sru = CustomSRU(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.sru = CustomSRU(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)
        else:
            raise ValueError("Task name not supported!")

    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        result = self.sru(x, target, criterion, model_optim, inp=inp, mask=mask)

        return result


class CustomSRU(nn.Module):
    """
    SPTT-Hybrid SRU.

    SRU recurrence keeps the original v_f / v_r pointwise recurrent terms from the SRU design,
    while the output uses a modernized tanh(c_t) + skip variant, which matches current
    BPTT-SRU implementation direction.

    SPTT compression is only applied to the INPUT-SIDE MATRIX GRADIENTS:
        w_x, w_f, w_r, and optionally w_skip

    Exact accumulation is used for:
        v_f, v_r, b_x, b_f, b_r, and optionally b_skip
    """
    def __init__(self, feature_dim, hidden_dim, krank, time_block_num, device, config):
        super(CustomSRU, self).__init__()
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
            # 4 means wx, wf, wr, wskip
            self.stacked_out_dim = 4 * hidden_dim
            self.has_skip_proj = True
        else:
            self.w_skip = None
            self.b_skip = None
            self.stacked_out_dim = 3 * hidden_dim
            self.has_skip_proj = False
            
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
            self.X_matrix_in,
            self.Sigma_in,
            self.Sigma_matrix_in,
            self.Delta_matrix_in,
        )
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def init_sptt_parameters(self):
        """
        Only one low-rank system for SRU input-side matrix gradients.
        """
        self.X_matrix_in = torch.randn(self.feature_dim, self.krank, device=self.device)
        self.Delta_matrix_in = torch.randn(self.stacked_out_dim, self.krank, device=self.device)
        self.Sigma_in = torch.randn(self.krank, device=self.device)
        self.Sigma_matrix_in = torch.diag(self.Sigma_in)
            
    def forward(self, x, targets=None, criterion=None, model_optim=None, inp=None, mask=None):
        global_param["time_for_batch"] = 0.0
        
        if self.task_name == 'imputation':
            x_new = x
            x = inp
        
        num_samples = x.size(0)
        seq_len = x.size(1)
        
        hx = torch.zeros(num_samples, self.hidden_dim, device=self.device)
        cx = torch.zeros(num_samples, self.hidden_dim, device=self.device)
        
        indices = np.linspace(0, seq_len, self.time_block_num + 1, dtype=int)
        sum_loss = 0.0
        
        if model_optim is not None:
            chunk_block = 0
            for i in range(self.time_block_num):
                accumulator.reset()
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
                        self.w_x, self.b_x,
                        self.w_f, self.v_f, self.b_f,
                        self.w_r, self.v_r, self.b_r,
                        self.w_skip, self.b_skip,
                        self.has_skip_proj,
                        self.hidden_dim,
                        self.feature_dim,
                        time_step, chunk_block
                    )
                
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
                    model_optim.zero_grad()
                    loss.backward()
                    model_optim.step()
                    
            if self.task_name == 'classification':
                return outputs, loss, sum_loss
            else:
                return loss, sum_loss
            
        elif model_optim is None and criterion is not None:
            time_step = 0
            accumulator.reset()
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
                hx, cx = SRUCellFunction.apply(
                        inputs,
                        cx,
                        self.w_x, self.b_x,
                        self.w_f, self.v_f, self.b_f,
                        self.w_r, self.v_r, self.b_r,
                        self.w_skip, self.b_skip,
                        self.has_skip_proj,
                        self.hidden_dim,
                        self.feature_dim,
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
            accumulator.reset()
            
            time_step = 0
            for i in range(x.size(1)):
                time_step += 1
                inputs = x[:, i, :]
            
            hx, cx = SRUCellFunction.apply(
                        inputs,
                        cx,
                        self.w_x, self.b_x,
                        self.w_f, self.v_f, self.b_f,
                        self.w_r, self.v_r, self.b_r,
                        self.w_skip, self.b_skip,
                        self.has_skip_proj,
                        self.hidden_dim,
                        self.feature_dim,
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
    @staticmethod
    def forward(
        ctx,
        inputs,
        cx,
        w_x, b_x,
        w_f, v_f, b_f,
        w_r, v_r, b_r,
        w_skip, b_skip,
        has_skip_proj,
        hidden_dim,
        feature_dim,
        time_step=None, chunk_block=None
    ):
        
        # transformed input
        x_hat = torch.mm(inputs, w_x.t()) + b_x
        
        # gates
        f_pre = torch.mm(inputs, w_f.t()) + b_f + cx * v_f
        r_pre = torch.mm(inputs, w_r.t()) + b_r + cx * v_r

        forgetgate = torch.sigmoid(f_pre)
        resetgate = torch.sigmoid(r_pre)

        # cell update
        cy = forgetgate * cx + (1.0 - forgetgate) * x_hat
        
        # skip
        if has_skip_proj:
            x_skip = torch.mm(inputs, w_skip.t()) + b_skip
        else:
            if feature_dim != hidden_dim:
                raise ValueError("feature_dim must equal hidden_dim when skip projection is disabled.")
            x_skip = inputs
            
        tanh_cy = torch.tanh(cy)
        hy = resetgate * tanh_cy + (1.0 - resetgate) * x_skip   
        
        ctx.has_skip_proj = has_skip_proj
        ctx.hidden_dim = hidden_dim
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block
        
        if has_skip_proj:
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

        return hy, cy

    @staticmethod
    def backward(ctx, grad_hy, grad_cy):
        has_skip_proj = ctx.has_skip_proj
        hidden_dim = ctx.hidden_dim
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block

        if has_skip_proj:
            (
                inputs, cx,
                x_hat, x_skip,
                forgetgate, resetgate,
                cy, tanh_cy,
                w_x, b_x, 
                w_f, v_f, b_f, 
                w_r, v_r, b_r, 
                w_skip, b_skip
            ) = ctx.saved_tensors
        else:
            (
                inputs, cx,
                x_hat, x_skip,
                forgetgate, resetgate,
                cy, tanh_cy,
                w_x, b_x, 
                w_f, v_f, b_f, 
                w_r, v_r, b_r
            ) = ctx.saved_tensors

        # ---- exact local backward for state/input propagation ----
        grad_resetgate = grad_hy * (tanh_cy - x_skip)
        grad_tanh_cy = grad_hy * resetgate
        grad_x_skip = grad_hy * (1.0 - resetgate)

        grad_cy_total = grad_tanh_cy * (1.0 - tanh_cy ** 2) + grad_cy

        grad_forgetgate = grad_cy_total * (cx - x_hat)
        grad_cx = grad_cy_total * forgetgate
        grad_x_hat = grad_cy_total * (1.0 - forgetgate)

        grad_f_pre = grad_forgetgate * forgetgate * (1.0 - forgetgate)
        grad_r_pre = grad_resetgate * resetgate * (1.0 - resetgate)

        grad_cx = grad_cx + grad_f_pre * v_f + grad_r_pre * v_r

        # exact vector grads
        grad_v_f = torch.sum(grad_f_pre * cx, dim=0)
        grad_v_r = torch.sum(grad_r_pre * cx, dim=0)

        # exact bias grads
        grad_b_x = grad_x_hat.sum(0)
        grad_b_f = grad_f_pre.sum(0)
        grad_b_r = grad_r_pre.sum(0)
        
        if has_skip_proj:
            grad_b_skip = grad_x_skip.sum(0)
        else:
            grad_b_skip = None

        # exact input gradient
        grad_inputs = (
            torch.mm(grad_x_hat, w_x) +
            torch.mm(grad_f_pre, w_f) +
            torch.mm(grad_r_pre, w_r)
        )
        if has_skip_proj:
            grad_inputs = grad_inputs + torch.mm(grad_x_skip, w_skip)
        else:
            grad_inputs = grad_inputs + grad_x_skip
        
        # ---- SPTT accumulation phase ----
        if has_skip_proj:
            delta_input = torch.cat((grad_x_hat, grad_f_pre, grad_r_pre, grad_x_skip), dim=1)
        else:
            delta_input = torch.cat((grad_x_hat, grad_f_pre, grad_r_pre), dim=1)

        accumulator.accumulate(
            input_act=inputs,
            delta_input=delta_input,
        )
            
        # ---- one-shot reconstruction phase ----
        grad_w_x = None
        grad_w_f = None
        grad_w_r = None
        grad_w_skip = None
        
        if time_step == 1:
            # global_param["flag"] = False

            X_matrix_in, Sigma_in, Sigma_matrix_in, Delta_matrix_in = accumulator.deliver_sptt_parameter()
            input_cat, delta_input_cat = accumulator.get_concatenated_gradients()

            i = chunk_block
            
            inv_Sigma_matrix_in = torch.inverse(Sigma_matrix_in)
            
            start_calculate_gradients = time.time()
            with torch.no_grad():
                activation_input = input_cat
                delta = delta_input_cat
                    
                current_t = activation_input.size(0)
                if current_t == 0:
                    raise ValueError("Current time block has zero samples, which should not happen.")
                
                scale_factor_right = delta @ Delta_matrix_in / current_t
                scale_factor_left = activation_input @ X_matrix_in / current_t
                
                i_factor = i / (i + 1)
                update_factor = 1.0 / (i + 1)
                
                X_update = activation_input.t() @ scale_factor_right @ inv_Sigma_matrix_in
                Delta_update = delta.t() @ scale_factor_left @ inv_Sigma_matrix_in
                
                X_matrix_in = i_factor * X_matrix_in + update_factor * X_update
                Delta_matrix_in = i_factor * Delta_matrix_in + update_factor * Delta_update
                
                X_matrix_in, _, sigma_signal_X = QR_matrix(X_matrix_in, X_matrix_in.shape)
                Delta_matrix_in, _, sigma_signal_D = QR_matrix(Delta_matrix_in, Delta_matrix_in.shape)
                
                align = sigma_signal_X * sigma_signal_D
                Delta_matrix_in = Delta_matrix_in * align.unsqueeze(0)
                
                
                Sigma_product = torch.sum(
                    ((delta @ Delta_matrix_in) * (activation_input @ X_matrix_in)) / ((i + 1) * current_t),
                    dim=0
                )
                
                Sigma_in = i_factor * Sigma_in + Sigma_product
                
                Sigma_matrix_in = torch.diag(Sigma_in)
                
                inv_Sigma_matrix_in = torch.inverse(Sigma_matrix_in)
                
            grad_w_in = (X_matrix_in @ Sigma_matrix_in @ Delta_matrix_in.t()).t()
            
            if has_skip_proj:
                grad_w_x = grad_w_in[0:hidden_dim, :]
                grad_w_f = grad_w_in[hidden_dim:2 * hidden_dim, :]
                grad_w_r = grad_w_in[2 * hidden_dim:3 * hidden_dim, :]
                grad_w_skip = grad_w_in[3 * hidden_dim:4 * hidden_dim, :]
            else:
                grad_w_x = grad_w_in[0:hidden_dim, :]
                grad_w_f = grad_w_in[hidden_dim:2 * hidden_dim, :]
                grad_w_r = grad_w_in[2 * hidden_dim:3 * hidden_dim, :]
                grad_w_skip = None
            
            end_calculate_gradients = time.time()
            global_param["time_for_batch"] = end_calculate_gradients - start_calculate_gradients

            accumulator.save_sptt_parameter(
                X_matrix_in, Sigma_in, Sigma_matrix_in, Delta_matrix_in
            )        
            
        return (
            grad_inputs,        # inputs
            grad_cx,            # cx
            grad_w_x, grad_b_x, # w_x, b_x
            grad_w_f, grad_v_f, grad_b_f,
            grad_w_r, grad_v_r, grad_b_r,
            grad_w_skip, grad_b_skip,
            None,               # has_skip_proj
            None,               # hidden_dim
            None,               # feature_dim
            None, None    # time_step, chunk_block
        )   