import os
import torch
import numpy as np
import logging
from datetime import datetime

class GradientWeightLogger:
    """
    用于监控和记录神经网络训练过程中的梯度和权重值
    可以帮助诊断NaN和梯度爆炸/消失问题
    """
    
    def __init__(self, model, log_dir='./logs', log_interval=1, 
                 log_histograms=False, max_batches_to_log=10):
        """
        初始化梯度和权重记录器
        
        参数:
            model: PyTorch模型
            log_dir: 日志文件保存目录
            log_interval: 每隔多少个batch记录一次
            log_histograms: 是否记录完整的直方图数据（会增加日志大小）
            max_batches_to_log: 达到此批次数后停止详细记录（防止日志过大）
        """
        self.model = model
        self.log_interval = log_interval
        self.log_histograms = log_histograms
        self.max_batches_to_log = max_batches_to_log
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logging.getLogger('gradient_weight_logger')
        self.logger.setLevel(logging.INFO)
        
        # 创建日志文件处理器
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'grad_weight_log_{timestamp}.txt')
        file_handler = logging.FileHandler(log_file)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 初始化批次计数器
        self.batch_count = 0
        
        # 记录模型结构
        self.logger.info(f"Model Architecture:\n{model}")
        
    def log_stats(self, batch_idx, loss_value=None):
        """
        记录当前批次的梯度和权重统计信息
        
        参数:
            batch_idx: 当前批次索引
            loss_value: 当前批次的损失值（可选）
        """
        self.batch_count += 1
        
        # 检查是否需要记录此批次
        if self.batch_count % self.log_interval != 0:
            return
            
        # 检查是否已超过最大批次记录数
        detailed_logging = self.batch_count <= self.max_batches_to_log
        
        # 记录批次信息和损失值
        log_prefix = f"Batch {batch_idx}"
        if loss_value is not None:
            # 确保损失值是CPU上的标量
            if isinstance(loss_value, torch.Tensor):
                loss_scalar = loss_value.item() if loss_value.numel() == 1 else loss_value.detach().cpu().mean().item()
            else:
                loss_scalar = loss_value
                
            self.logger.info(f"{log_prefix} - Loss: {loss_scalar:.6f}")
            
            # 检查损失是否为NaN或Inf
            if isinstance(loss_value, torch.Tensor) and (torch.isnan(loss_value).any() or torch.isinf(loss_value).any()):
                self.logger.warning(f"{log_prefix} - ⚠️ Loss contains {'NaN' if torch.isnan(loss_value).any() else 'Inf'}!")
        
        # 遍历所有参数记录梯度和权重信息
        for name, param in self.model.named_parameters():
            # 记录权重信息
            weight_stats = self._compute_tensor_stats(param.data, "Weight")
            self.logger.info(f"{log_prefix} - {name} - {weight_stats}")
            
            # 记录梯度信息（如果存在）
            if param.grad is not None:
                grad_stats = self._compute_tensor_stats(param.grad, "Grad")
                self.logger.info(f"{log_prefix} - {name} - {grad_stats}")
                
                # 检查梯度是否为NaN或Inf
                grad_has_nan = torch.isnan(param.grad).any()
                grad_has_inf = torch.isinf(param.grad).any()
                
                if grad_has_nan or grad_has_inf:
                    self.logger.warning(f"{log_prefix} - {name} - ⚠️ Contains {'NaN' if grad_has_nan else 'Inf'} gradients!")
                    
                    # 详细记录首个NaN或Inf梯度的位置
                    if detailed_logging:
                        self._log_problem_indices(param.grad, name, "gradient")
            
            # 检查权重是否为NaN或Inf
            weight_has_nan = torch.isnan(param.data).any()
            weight_has_inf = torch.isinf(param.data).any()
            
            if weight_has_nan or weight_has_inf:
                self.logger.warning(f"{log_prefix} - {name} - ⚠️ Contains {'NaN' if weight_has_nan else 'Inf'} values!")
                
                # 详细记录首个NaN或Inf权重的位置
                if detailed_logging:
                    self._log_problem_indices(param.data, name, "weight")
                    
            # 记录直方图数据（可选）
            if detailed_logging and self.log_histograms:
                if param.grad is not None:
                    self._log_histogram(param.grad, name, "gradient")
                self._log_histogram(param.data, name, "weight")
    
    def _compute_tensor_stats(self, tensor, prefix):
        """计算张量的统计信息"""
        if tensor.numel() == 0:
            return f"{prefix} stats: empty tensor"
            
        with torch.no_grad():
            # 将张量转换为CPU并展平
            flat_tensor = tensor.detach().cpu().flatten()
            
            # 计算统计量
            tensor_min = flat_tensor.min().item()
            tensor_max = flat_tensor.max().item()
            tensor_mean = flat_tensor.mean().item()
            tensor_std = flat_tensor.std().item()
            tensor_norm = torch.norm(flat_tensor).item()
            
            # 检查是否有NaN或Inf
            has_nan = torch.isnan(flat_tensor).any().item()
            has_inf = torch.isinf(flat_tensor).any().item()
            
            # 记录其他有用的指标
            zeros_pct = (flat_tensor == 0).float().mean().item() * 100
            
            return (f"{prefix} stats: min={tensor_min:.6e}, max={tensor_max:.6e}, "
                    f"mean={tensor_mean:.6e}, std={tensor_std:.6e}, norm={tensor_norm:.6e}, "
                    f"zeros={zeros_pct:.2f}%, NaN={has_nan}, Inf={has_inf}")
    
    def _log_problem_indices(self, tensor, name, tensor_type):
        """记录张量中首个问题值的索引"""
        with torch.no_grad():
            # 将张量移至CPU
            cpu_tensor = tensor.detach().cpu()
            
            # 找出NaN索引
            nan_mask = torch.isnan(cpu_tensor)
            if nan_mask.any():
                nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                if len(nan_indices) > 0:
                    first_nan = nan_indices[0].cpu().numpy()
                    self.logger.warning(f"First NaN {tensor_type} in {name} at index: {first_nan}")
                
            # 找出Inf索引
            inf_mask = torch.isinf(cpu_tensor)
            if inf_mask.any():
                inf_indices = torch.nonzero(inf_mask, as_tuple=False)
                if len(inf_indices) > 0:
                    first_inf = inf_indices[0].cpu().numpy()
                    self.logger.warning(f"First Inf {tensor_type} in {name} at index: {first_inf}")
    
    def _log_histogram(self, tensor, name, tensor_type):
        """记录张量数值分布的直方图数据"""
        with torch.no_grad():
            # 先将整个张量移到CPU
            cpu_tensor = tensor.detach().cpu()
            
            # 然后在CPU上进行有限值检查
            is_finite = torch.isfinite(cpu_tensor)
            finite_tensor = cpu_tensor[is_finite]
            
            if finite_tensor.numel() == 0:
                self.logger.warning(f"No finite values to create histogram for {name} {tensor_type}")
                return
                
            # 计算直方图
            try:
                hist_values, hist_bins = torch.histogram(finite_tensor.flatten(), bins=10)
                hist_values = hist_values.numpy()
                hist_bins = hist_bins.numpy()
                
                # 生成直方图字符串
                hist_str = "Histogram bins: "
                for i in range(len(hist_bins)-1):
                    hist_str += f"[{hist_bins[i]:.2e},{hist_bins[i+1]:.2e}]:{hist_values[i]} "
                
                self.logger.info(f"{name} {tensor_type} {hist_str}")
            except Exception as e:
                self.logger.warning(f"Failed to compute histogram for {name} {tensor_type}: {e}")