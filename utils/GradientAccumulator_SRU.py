import torch
from global_param import global_param

class GradientAccumulatorSRU_Hybrid:
    """
    Accumulate SRU activations/errors for SPTT-Hybrid.

    For SRU input-side matrix gradients:
        grad_W = delta^T @ input

    We therefore accumulate:
        - input activations
        - stacked input-side error matrix delta_input

    And we exact-accumulate:
        - vector gradients: v_f, v_r
        - bias gradients: b_x, b_f, b_r, (optional) b_skip
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.inputs = []
        self.delta_inputs = []
        self.length = 0
        
    def accumulate(
        self,
        input_act: torch.Tensor,
        delta_input: torch.Tensor,
    ):
        self.inputs.append(input_act.detach())
        self.delta_inputs.append(delta_input.detach())
        
        # if grad_b_skip is not None:
        #     self.grad_b_skip.append(grad_b_skip.detach())
        
        self.length += input_act.size(0)
        
        # if self.length >= global_param["total_time_block_size"]:
        #     global_param["compression_finished"] = True
            
    def get_concatenated_gradients(self):
        return (
            torch.cat(self.inputs, dim=0),
            torch.cat(self.delta_inputs, dim=0),
        )
        
    def save_sptt_parameter(
        self, 
        X_matrix_in,
        Sigma_in,
        Sigma_matrix_in,
        Delta_matrix_in,
    ):
        self.X_matrix_in = X_matrix_in
        self.Sigma_in = Sigma_in
        self.Sigma_matrix_in = Sigma_matrix_in
        self.Delta_matrix_in = Delta_matrix_in
        
    def deliver_sptt_parameter(self):
        return (
            self.X_matrix_in,
            self.Sigma_in,
            self.Sigma_matrix_in,
            self.Delta_matrix_in,
        )
    
    def save_time_backward(self, time_backward):
        self.time_backward = time_backward
        
    def deliver_time_backward(self):
        return self.time_backward