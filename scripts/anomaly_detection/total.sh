#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

anomaly_ratio=1

seed=2026
############################## SWAT ########################
# data_name=SWAT
# seq_len=100
# batch_size=256

# echo "开始运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

# ./scripts/anomaly_detection/Bptt/BpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Sptt/SpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# ./scripts/anomaly_detection/Sptt/SpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Bptt/BpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# echo "结束运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"


############################# MSL ########################
# data_name=MSL
# seq_len=100
# batch_size=128

# echo "开始运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

# ./scripts/anomaly_detection/Bptt/BpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Sptt/SpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# ./scripts/anomaly_detection/Sptt/SpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Bptt/BpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# echo "结束运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

# ############################# PSM ########################
# data_name=PSM
# seq_len=100
# batch_size=128

# echo "开始运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

# ./scripts/anomaly_detection/Bptt/BpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Sptt/SpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# ./scripts/anomaly_detection/Sptt/SpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Bptt/BpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# echo "结束运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"



############################## SMD ########################
# data_name=SMD
# seq_len=100
# batch_size=128

# echo "开始运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

# ./scripts/anomaly_detection/Bptt/BpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Sptt/SpttLSTM.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# ./scripts/anomaly_detection/Sptt/SpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"
# ./scripts/anomaly_detection/Bptt/BpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"

# echo "结束运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"



############################## SMAP ########################
data_name=SMAP
seq_len=100
batch_size=128

echo "开始运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

./scripts/anomaly_detection/Sptt/SpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"


echo "结束运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

data_name=SWAT

echo "开始运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"

./scripts/anomaly_detection/Sptt/SpttGRU.sh "$data_name" "$seed" "$seq_len" "$anomaly_ratio" "$batch_size"


echo "结束运行 $data_name, 序列长度 $seq_len, 异常比例 $anomaly_ratio, 随机种子 $seed"
