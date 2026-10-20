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

## 6. Delayed Copying Task

To investigate whether SPTT can mitigate the difficulties of learning long-range dependencies under truncated backpropagation, we introduce a **delayed copying task with strict endpoint supervision**.

The task requires an LSTM to retain information presented at the beginning of a sequence and reproduce it at the final time step. The delay exceeds the backward horizon of a single time block, allowing us to evaluate learning performance under truncated gradient propagation.

Further experimental details and results are provided in the Supplementary Information of our paper.

### 6.1 Model Implementations

We provide four implementations for comparison:

| Method | Implementation | Description |
|---|---|---|
| Full BPTT | `models/Bptt/FullBpttLSTM_Delay.py` | Standard BPTT without temporal truncation |
| TBPTT | `models/Bptt/BpttLSTM_Delay.py` | Truncated BPTT with detached hidden states at block boundaries |
| SPTT-Window | `models/Sptt/SpttLSTM_Delay.py` | SPTT with truncated error signals and inherited low-rank gradient states |
| SPTT-Bounded | `models/Sptt/SpttLSTM_Delay_Bounded.py` | SPTT-Window with bounded spectral coefficients for numerical stability |

All four implementations use the same LSTM architecture and delayed-copying task protocol.

For both TBPTT and SPTT-Window, hidden and cell states remain continuous during forward propagation, while their computational graphs are detached at block boundaries.

SPTT additionally retains compressed historical gradient information through inherited low-rank factors. This mechanism does not reconstruct the exact temporal derivatives removed by truncation.

### 6.2 Dataset Generation

We provide a dataset generator at:

`data_provider/delayed_copy_dataset.py`

Each input sequence consists of \(T\) time steps, with a **16-dimensional feature vector** at each step.

The dataset is constructed as follows:

1. **Initial cue:** At the first time step, a class \(c\) is randomly selected from \(C\) possible classes. The corresponding input dimension is set to 1, forming a one-hot cue.
2. **Intermediate steps:** The remaining intermediate steps contain either zero padding or label-independent Gaussian noise in designated distractor dimensions.
3. **Final query:** At the final time step, the 16th input dimension is set to 1 to indicate a query.
4. **Prediction target:** The model must reproduce the class presented at the first time step.

A compact representation of the task is:

$$
x_1=e_c,\qquad
x_t=\epsilon_t\quad (1<t<T),\qquad
x_T=e_{16},\qquad y=c,
$$

where $e_c$ is a 16-dimensional one-hot vector and $\epsilon_t$ represents zero padding or label-independent distractor noise.

**Strict endpoint supervision:** The classification loss is computed only at the final time step. No intermediate losses are introduced.

This ensures that the truncated methods cannot backpropagate the terminal loss into the early blocks containing the class cue.

#### Generate a dataset

From the repository root, run:

```bash
cd data_provider

python delayed_copy_dataset.py \
    --seq-len 100 \
    --num-classes 15 \
    --noise-std 1.0 \
    --seed 2026
```

The parameters control:

| Parameter | Description |
|---|---|
| `--seq-len` | Total sequence length \(T\) |
| `--num-classes` | Number of target classes \(C\) |
| `--noise-std` | Standard deviation of the intermediate Gaussian noise |
| `--seed` | Random seed used for dataset generation |

For the experiments reported in our response to the reviewer, we used a dataset-generation seed of `2026` and independent training seeds from `2023` to `2028`.

The training, validation, and test sets are generated independently. Model checkpoints are selected using validation performance, while the test set is reserved for final evaluation.

### 6.3 Running the Experiments

We provide experiment scripts under:

`scripts/delayed_copy/`

The script `total_delay.sh` is intended to run the configured delayed-copying experiments.

From the repository root, execute:

```bash
bash scripts/delayed_copy/total_delay.sh
```

Alternatively, users can modify the corresponding scripts to run individual methods or experimental configurations.

The main experimental settings are summarized below:

| Sequence length \(T\) | Number of blocks | Classes | Noise std. |
|---|---:|---:|---:|
| 10 | 2 | 8 | 0.0 |
| 25 | 2 | 8 | 0.0 |
| 50 | 5 | 8 | 0.0 |
| 100 | 5 | 15 | 1.0 |
| 200 | 10 | 15 | 1.0 |
| 400 | 20 | 15 | 1.0 |
| 600 | 30 | 15 | 1.0 |

The original SPTT-Window implementation is evaluated on the shorter configurations, while the longer configurations are used to evaluate SPTT-Bounded.

For \(T=200,400,600\), the backward horizon is fixed at **20 time steps**, even though the total sequence length increases substantially.

### 6.4 SPTT-Specific Settings

**Historical low-rank inheritance.** In the strict endpoint-supervised task, only the final block contributes recurrent-gradient information. Therefore, SPTT retains its low-rank state across successive mini-batch updates rather than receiving gradients from the earlier, unsupervised blocks of the same sequence.

For this experiment, the weighting coefficient is indexed by the cumulative mini-batch update count \(n\):

$$
\rho_n=\frac{n}{n+1},\qquad
1-\rho_n=\frac{1}{n+1}.
$$

This task-specific indexing differs from the within-call block indexing used in the main SPTT experiments.

**Spectral stabilization.** SPTT-Bounded additionally constrains the retained spectral coefficients:

$$
\widetilde{\sigma}_j = \operatorname{sign}(\sigma_j) \min\left\{10^4,\max(10^{-4},|\sigma_j|) \right\}.
$$

The remaining low-rank tracking and inheritance mechanisms are retained.

**Checkpoint selection.** Models are evaluated using their best valid checkpoints selected on the validation set. If non-finite values are encountered during training, training is terminated and the best previously saved valid checkpoint is used for evaluation, where available. Such termination should not be interpreted as numerically stable convergence.

### 6.5 Reproducibility

All reported experiments use six independent model initialization seeds:

```text
2023, 2024, 2025, 2026, 2027, 2028
```

The task is designed to evaluate whether recurrent models can learn delayed dependencies under different gradient propagation and low-rank tracking mechanisms.

Importantly, successful delayed copying under truncation does not imply recovery of exact cross-boundary derivatives. Instead, the experiment evaluates the practical ability of SPTT-based optimization to learn long-range dependencies despite an explicitly limited backward horizon.
