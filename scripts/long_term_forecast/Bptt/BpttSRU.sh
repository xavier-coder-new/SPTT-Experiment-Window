#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=BpttSRU
data_name=$1
seed=$2
features=M
train_epochs=300
patience=10
time_block_num=$3
krank=$4
learning_rate=$5
inverse_model=$6

if [[ "$data_name" = "ETTh1" || "$data_name" = "ETTh2" || "$data_name" = "ETTm1" || "$data_name" = "ETTm2" ]]; then
    data_dict=$data_name
    root_path=./dataset/ETT-small/
    feature_dim=7
    if [[ "$data_name" = "ETTh1" || "$data_name" = "ETTh2" ]]; then
        batch_size=64
        seq_len=384
        pred_len=96
        label=384_96
    elif [[ "$data_name" = "ETTm1" || "$data_name" = "ETTm2" ]]; then
        batch_size=256
        seq_len=480
        pred_len=192
        label=480_192
    fi
else
    data_dict=custom
    root_path="./dataset/$data_name/"
    if [[ "$data_name" = "traffic" ]]; then
        feature_dim=862
        batch_size=64
        seq_len=480
        pred_len=192
        label=480_192
    elif [[ "$data_name" = "electricity" ]]; then
        feature_dim=321
        batch_size=128
        seq_len=672
        pred_len=384
        label=672_384
    elif [[ "$data_name" = "weather" ]]; then
        feature_dim=21
        batch_size=128
        seq_len=672
        pred_len=384
        label=672_384
    elif [[ "$data_name" = "illness" ]]; then
        patience=30
        feature_dim=7
        batch_size=32
        seq_len=96
        pred_len=96
        label=96_96
    elif [[ "$data_name" = "exchange_rate" ]]; then
        feature_dim=8
        batch_size=64
        seq_len=192
        pred_len=96
        label=192_96
    fi
fi

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id $data_name''_"$label" \
    --model $model_name \
    --seed $seed \
    --data $data_dict \
    --data_name $data_name \
    --root_path "$root_path" \
    --data_path $data_name.csv \
    --features $features \
    --krank $krank \
    --time_block_num $time_block_num \
    --feature_dim "$feature_dim" \
    --itr 1 \
    --train_epochs $train_epochs \
    --batch_size "$batch_size" \
    --patience "$patience" \
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
    --estimation_method None \
    --inverse_model $inverse_model


