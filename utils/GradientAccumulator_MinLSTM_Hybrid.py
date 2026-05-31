import torch
from global_param import global_param

class GradientAccumulator_MinLSTM_Hybrid:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.inputs = []
        self.delta_ih = []
        self.length = 0
        self.time_back_ward = 0.0
        
    def accumulate(self, inputs, delta_ih):
        self.inputs.append(inputs)
        self.delta_ih.append(delta_ih)
        self.length += len(inputs)

        # if self.length >= global_param['total_time_block_size']:
        #     global_param['compression_finished'] = True
            
    def get_concatenated_gradients(self):
        return (
            torch.cat(self.inputs, dim=0),
            torch.cat(self.delta_ih, dim=0),
        )
        
    def save_sptt_parameter(self, X_matrix_ih, Sigma_ih, Sigma_matrix_ih, Delta_matrix_ih):
        self.X_matrix_ih = X_matrix_ih
        self.Sigma_ih = Sigma_ih
        self.Sigma_matrix_ih = Sigma_matrix_ih
        self.Delta_matrix_ih = Delta_matrix_ih
        
    def deliver_sptt_parameter(self):
        return (
            self.X_matrix_ih,
            self.Sigma_ih,
            self.Sigma_matrix_ih,
            self.Delta_matrix_ih,
        ) 
        
    def save_time_backward(self, time_backward):
        self.time_back_ward = time_backward

    def deliver_time_backward(self):
        return self.time_back_ward