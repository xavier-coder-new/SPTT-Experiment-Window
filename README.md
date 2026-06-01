# Streaming Propagation Through Time for Training Recurrent Neural Networks

This repository provides the implementation of the **SPTT-Window** variant of the **Streaming Propagation Through Time (SPTT)** framework proposed in our paper *"Streaming Propagation Through Time for Training Recurrent Neural Networks."*

---

## 1. Code Structure and Naming Convention

The codebase contains two main implementations:

- Files named **`BPTT`** correspond to the **TBPTT (Truncated Backpropagation Through Time)** implementation.  
- Files named **`SPTT`** correspond to the **SPTT-Window** implementation.

Both implementations follow the same data loading and training interface, allowing a direct comparison under identical experimental settings.

---

## 2. Parameter Configuration

The number of **time blocks** used in the SPTT-Window framework can be adjusted via the `time_block_num` parameter.  
Please modify this parameter in the corresponding task script to control how the input sequence is divided into temporal segments.

---

## 3. Reference Framework

Our implementation is developed based on the [Time-Series-Library](https://github.com/thuml/Time-Series-Library). All other configurations (e.g., dataset structure, training procedure, environment setup) are consistent with those described in the main SPTT repository.

---

## 4. Experimental Environment

All experiments were conducted on the following hardware and software configuration:

- GPU: NVIDIA RTX 4090
- CPU: Intel Core i7-14700KF
- PyTorch: 2.1.0
- Python: 3.8.19
- Pytorch-cuda: 11.8

We provided the yaml file of our experimental environment, and you can use conda to configure according environment:
```bash
conda env create -f sptt_main.yaml
```
---

## 5. Visualization

We provide one script to visualize the training dynamics of the models:
- `utils/plot_time_block.py`: This script generates a comprehensive figure that shows the training dynamics of the models for different time blocks.

⚽️Note: We specifically monitor the loss curves for two representative tasks: long_term_forecast and classification. If you wish to monitor the training process for these specific tasks, you need to modify the main function interface in run.py.

- For the long-term forecasting task, replace Exp = Exp_Long_Term_Forecast with `Exp = Exp_Long_Term_Forecast_Plot_Loss`.
- For the classification task, replace Exp = Exp_Classification with `Exp = Exp_Classification_Plot_Loss`.
