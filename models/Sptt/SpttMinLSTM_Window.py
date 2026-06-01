import torch
import torch.nn as nn
import math
from global_param import global_param
import time
from utils.GradientAccumulator_MinLSTM import GradientAccumulator_MinLSTM_Hybrid as GradientAccumulator
from icecream import ic
import numpy as np

accumulator =  GradientAccumulator()

def QR_matrix(X_matrix, target_shape):
    Q, R, diagonal_sign = qr_with_non_negative_diagonal(X_matrix)

    return Q, R, diagonal_sign


def qr_with_non_negative_diagonal(matrix):
    Q, R = torch.linalg.qr(matrix)

    diagonal_sign = torch.sign(torch.diag(R))
    
    diagonal_sign[diagonal_sign == 0] = 1

    Q = Q * diagonal_sign.unsqueeze(0)
    
    R = diagonal_sign.unsqueeze(1) * R

    return Q, R, diagonal_sign


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
            self.lstm = CustomMinLSTM(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)

        elif self.task_name == 'classification':
            self.output_size = config.num_classes
            self.feature_dim = config.feature_dim
            self.lstm = CustomMinLSTM(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.output_size)

        elif self.task_name == 'imputation':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomMinLSTM(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)

        elif self.task_name == 'anomaly_detection':
            self.output_size = config.seq_len
            self.feature_dim = config.feature_dim
            self.lstm = CustomMinLSTM(
                self.feature_dim, self.hidden_dim, self.krank, self.time_block_num, self.device, config
            )
            self.fc = nn.Linear(self.hidden_dim, self.feature_dim * self.output_size)

        else:
            raise ValueError("Task name not supported!")

    def forward(self, x, target=None, criterion=None, model_optim=None, inp=None, mask=None):
        result = self.lstm(x, target, criterion, model_optim, inp=inp, mask=mask)
        
        return result
    
    
class CustomMinLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, krank, time_block_num, device, config):
        super(CustomMinLSTM, self).__init__()
        
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
        
        # minLSTM:
        # f_t = sigmoid(W_f x_t + b_f)
        # i_t = sigmoid(W_i x_t + b_i)
        # h_tilde = W_h x_t + b_h
        self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, feature_dim))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
        
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
        std = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
            
    def init_sptt_parameters(self):
        self.X_matrix_ih = torch.randn(self.feature_dim, self.krank).to(self.device)
        self.Sigma_ih = torch.randn(self.krank).to(self.device)
        self.Sigma_matrix_ih = torch.diag(self.Sigma_ih).to(self.device)
        self.Delta_matrix_ih = torch.randn(3 * self.hidden_dim, self.krank).to(self.device) 
        
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
                
                time_step = 0
                
                for i in range(chunk_start, chunk_end):
                    time_step += 1
                    inputs = x[:, i, :]
                    hx = MinLSTMCellFunction.apply(
                        inputs, hx, self.w_ih, self.b_ih, time_step, chunk_block
                    )
                
                fc_out = self.fc(hx)
                
                if self.task_name == "long_term_forecast":
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
    def forward(ctx, inputs, hidden_state, w_ih, b_ih, time_step=None, chunk_block=None):
        gates = torch.mm(inputs, w_ih.t()) + b_ih
        forgetgate_pre, inputgate_pre, h_tilde = gates.chunk(3, 1)
        
        forgetgate = torch.sigmoid(forgetgate_pre)
        inputgate = torch.sigmoid(inputgate_pre)
        
        denom = forgetgate + inputgate + 1e-8
        f_prime = forgetgate / denom
        i_prime = inputgate / denom
        
        hy = f_prime * hidden_state + i_prime * h_tilde
        
        ctx.save_for_backward(
            inputs, hidden_state, forgetgate, inputgate, h_tilde, f_prime, i_prime, denom, w_ih
        )
        
        ctx.time_step = time_step
        ctx.chunk_block = chunk_block

        return hy
    
    @staticmethod
    def backward(ctx, grad_hy):                
        inputs, hidden_state, forgetgate, inputgate, h_tilde, f_prime, i_prime, denom, w_ih = ctx.saved_tensors
        time_step = ctx.time_step
        chunk_block = ctx.chunk_block

        d_input = d_hidden_state = grad_w_ih = grad_b_ih = None
        
        # hy = f' * h_prev + i' * h_tilde
        d_hidden_state = grad_hy * f_prime
        d_h_tilde = grad_hy * i_prime

        # gradients of normalized gates
        d_f_prime = grad_hy * hidden_state
        d_i_prime = grad_hy * h_tilde

        s2 = denom * denom

        # f' = f / (f + i), i' = i / (f + i)
        d_forgetgate = (inputgate / s2) * (d_f_prime - d_i_prime)
        d_inputgate = (forgetgate / s2) * (d_i_prime - d_f_prime)

        d_forgetgate_pre = d_forgetgate * forgetgate * (1 - forgetgate)
        d_inputgate_pre = d_inputgate * inputgate * (1 - inputgate)

        delta_ih = torch.cat((d_forgetgate_pre, d_inputgate_pre, d_h_tilde), dim=1)
        accumulator.accumulate(inputs, delta_ih)

        d_input = torch.mm(delta_ih, w_ih)
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
                
                Sigma_ih_product = torch.sum(
                    ((delta_ih @ Delta_matrix_ih) * (activation_input @ X_matrix_ih)) / ((i + 1) * current_t),
                    dim=0
                )
                Sigma_ih = i_factor * Sigma_ih + Sigma_ih_product                                                                                                 
                
                Sigma_matrix_ih = torch.diag(Sigma_ih)
                
                inv_Sigma_matrix_ih = torch.inverse(Sigma_matrix_ih)
                    
            grad_w_ih = (X_matrix_ih @ Sigma_matrix_ih @ Delta_matrix_ih.t()).t()
            
            end_calculate_gradients = time.time()
            global_param['time_for_batch'] = end_calculate_gradients - start_calculate_gradients
                
            accumulator.save_sptt_parameter(
                X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih
            )
            
        return d_input, d_hidden_state, grad_w_ih, grad_b_ih, None, None                                      
            
        