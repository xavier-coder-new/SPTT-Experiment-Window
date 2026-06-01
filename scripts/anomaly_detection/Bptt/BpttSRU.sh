#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=BpttSRU
data_name=$1
seed=$2
time_block_num=$3
krank=$4
learning_rate=$5
inverse_model=$6
features=M
train_epochs=100
pred_len=0
patience=10

anomaly_ratio=1

root_path=./dataset/"$data_name"/

if [[ "$data_name" == "MSL" ]]; then
    batch_size=128
    seq_len=100
    feature_dim=55
elif [[ "$data_name" == "PSM" ]]; then
    batch_size=128
    seq_len=100
    feature_dim=25
elif [[ "$data_name" == "SMAP" ]]; then
    batch_size=128
    seq_len=100
    feature_dim=25
elif [[ "$data_name" == "SMD" ]]; then
    batch_size=128
    seq_len=100
    feature_dim=38
elif [[ "$data_name" == "SWAT" ]]; then
    batch_size=256
    seq_len=100
    feature_dim=51
fi

label=$seq_len'_'$pred_len

python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --model_id "$data_name'_'$label" \
    --model $model_name \
    --seed $seed \
    --data "$data_name" \
    --data_name "$data_name" \
    --root_path "$root_path" \
    --data_path "$data_name".csv \
    --features $features \
    --krank $krank \
    --time_block_num $time_block_num \
    --feature_dim "$feature_dim" \
    --itr 1 \
    --train_epochs $train_epochs \
    --batch_size "$batch_size" \
    --patience $patience \
    --learning_rate $learning_rate \
    --loss MSE \
    --lradj type3 \
    --tmax 10 \
    --use_gpu True \
    --gpu 0 \
    --gpu_type cuda \
    --hidden_dim 512 \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --anomaly_ratio "$anomaly_ratio" \
    --estimation_method None \
    --inverse_model $inverse_model \
