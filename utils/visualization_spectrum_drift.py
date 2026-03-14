import os
import numpy as np
from global_param import global_param
import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic

def export_sigma_list_to_excel(save_dir, sigma_list, batch_num, patience, file_name="sigma_list.xlsx"):
    os.makedirs(save_dir, exist_ok=True)
    if sigma_list is None or len(sigma_list) == 0:
        return None

    sigma_array = np.asarray(sigma_list, dtype=np.float32)
    # ic("sigma_array shape:", sigma_array.shape)
    
    columns = [f"sigma_{idx + 1}" for idx in range(sigma_array.shape[1])]
    sigma_df = pd.DataFrame(sigma_array, columns=columns)
    sigma_df.insert(0, "batch_idx", np.arange(len(sigma_df)))

    absolute_spectrum_drifts = [np.nan]
    relative_spectrum_drifts = [np.nan]
    absolute_energy_retentions = [np.nan]
    
    for i in range(1, len(sigma_array)):
        drift = np.linalg.norm(sigma_array[i] - sigma_array[i - 1])
        base_norm = np.linalg.norm(sigma_array[i - 1])
        absolute_spectrum_drifts.append(drift)
        # relative_spectrum_drifts.append(drift / base_norm if base_norm > 1e-10 else 0.0)
        relative_spectrum_drifts.append(drift / base_norm)
        absolute_energy_retentions.append(np.linalg.norm(sigma_array[i]))

    sigma_df["abs_drift"] = absolute_spectrum_drifts
    sigma_df["rel_drift"] = relative_spectrum_drifts
    sigma_df["abs_energy_retention"] = absolute_energy_retentions
    
    sigma_trunc_df = sigma_df.iloc[:len(sigma_array) - batch_num * patience]
    
    excel_path = os.path.join(save_dir, file_name)
    sigma_df.to_excel(excel_path, index=False)
    sigma_trunc_df.to_excel(os.path.join(save_dir, f"truncated_{file_name}"), index=False)
    return excel_path


def plot_spectrum_drift(save_path, sigma_list, batch_num, patience, model, dpi=600):
    """
    Plot the spectrum drift across batches.
    Expects global_param['sigma_list'] to be a list of truncated singular value arrays.
    
    spectrum drift:
        - absolute spectrum drift: ||Xt - Xt-1|| L2 norm of the truncated singular values across batches.
        - relative spectrum drift: ||Xt - Xt-1|| / ||Xt-1|| L2 norm of the truncated singular values across batches.
    
    energy retention:
        - absolute energy retention: ||Xt|| L norm 
        - Need to note, sigma_list shape is (iteration, krank). For the current batch, regardless of the batch size, 
        there is only one gradient matrix and one weight matrix for each layer, with a shape of (feature_dim, feature_dim),
        and there is no dimension of "batch_num". So the Frobenius norm is not suitable for this case, which requests a two-dimensional array or matrix.
    
    
    Saves:
    - spectrum_drift.svg
"""
    export_sigma_list_to_excel(save_path, sigma_list, batch_num, patience)

    absolute_spectrum_drifts = []
    relative_spectrum_drifts = []
    absolute_energy_retention = []
    
    if model in {"BpttLSTM", "BpttGRU"}:
        index_start = 1
    else:
        index_start = 20
    
    # exclude the early stopping phase
    for i in range(index_start, len(sigma_list) - (batch_num * patience)):
        drift = np.linalg.norm(sigma_list[i] - sigma_list[i-1])
        absolute_spectrum_drifts.append(drift)
        # if np.linalg.norm(sigma_list[i-1]) > 1e-10:
        #     relative_spectrum_drifts.append(drift / np.linalg.norm(sigma_list[i-1]))
        # else:
        #     relative_spectrum_drifts.append(0)
        relative_spectrum_drifts.append(drift / np.linalg.norm(sigma_list[i-1]))

        # recode absolute energy retention
        # absolute_energy_retention.append(np.linalg.norm(sigma_list[i], ord='fro'))
        absolute_energy_retention.append(np.linalg.norm(sigma_list[i]))
        
    # Create subplot for truncated phase
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=dpi)
    
    ax1.plot(absolute_spectrum_drifts, label="Absolute Spectrum Drift", color='red')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Spectrum Drift")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(absolute_energy_retention, label="Absolute Energy Retention", color='purple')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Energy Retention (Frobenius Norm)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(save_path, "spectrum_drift_trunc.svg")
    plt.savefig(fig_path, dpi=dpi)
        
    # include the early stopping phase
    for i in range(len(sigma_list) - (batch_num * patience), len(sigma_list)):
        drift = np.linalg.norm(sigma_list[i] - sigma_list[i-1])
        absolute_spectrum_drifts.append(drift)
        relative_spectrum_drifts.append(drift / np.linalg.norm(sigma_list[i-1]))
        # absolute_energy_retention.append(np.linalg.norm(sigma_list[i], ord='fro'))
        absolute_energy_retention.append(np.linalg.norm(sigma_list[i]))
        
    # Create subplot for all iterations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=dpi)
    
    ax1.plot(absolute_spectrum_drifts, label="Absolute Spectrum Drift", color='blue')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Spectrum Drift")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(absolute_energy_retention, label="Absolute Energy Retention", color='magenta')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Energy Retention (Frobenius Norm)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    file_path = os.path.join(save_path, "spectrum_drift_all.svg")
    plt.savefig(file_path, dpi=dpi)
