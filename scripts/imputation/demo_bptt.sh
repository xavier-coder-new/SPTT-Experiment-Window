#!/bin/bash

model_name=BpttLSTM
data_name=ETTm2
seed=2025
time_block_num=8
learning_rate=0.001
inverse_model=align
mask_rate=0.25
features=M
train_epochs=300
patience=10
krank=7

if [[ "$data_name" = "ETTh1" || "$data_name" = "ETTh2" || "$data_name" = "ETTm1" || "$data_name" = "ETTm2" ]]; then
    data_dict=$data_name
    root_path=./dataset/ETT-small/
    feature_dim=7
    if [[ "$data_name" = "ETTh1" ]]; then
    # 这里需要注意一点，在做imputation（缺失值补全）任务时，通常输入长度seq_len和输出长度pred_len
    # 要设置为一样的，因为输入和输出都是完整的序列，而不是预测未来值。
        echo "使用ETTh1数据集"
        batch_size=64
        seq_len=96
        pred_len=96
        label=96_96
    elif [[ "$data_name" = "ETTh2" ]]; then
        echo "使用ETTh2数据集"
        batch_size=64
        seq_len=192
        pred_len=192
        label=192_192
    elif [[ "$data_name" = "ETTm1" ]]; then
        echo "使用ETTm1数据集"
        batch_size=128
        seq_len=288
        pred_len=288
        label=288_288
    elif [[ "$data_name" = "ETTm2" ]]; then
        echo "使用ETTm2数据集"
        batch_size=128
        seq_len=384
        pred_len=384
        label=384_384
    fi
else 
    data_dict=custom
    root_path="./dataset/$data_name/"
    if [[ "$data_name" = "electricity" ]]; then
        echo "使用electricity数据集"
        feature_dim=321
        batch_size=128
        seq_len=480
        pred_len=480
        label=480_480
    elif [[ "$data_name" = "weather" ]]; then
        echo "使用weather数据集"
        feature_dim=21
        batch_size=128
        seq_len=576
        pred_len=576
        label=576_576
    fi
fi

# maks_rate 0.125 0.25 0.375 0.5
python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --model_id "$data_name'_'$label" \
    --model $model_name \
    --seed $seed \
    --data "$data_dict" \
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
    --mask_rate "$mask_rate" \
    --inverse_model $inverse_model \
    --viz_spectrum_drift \
    # --record \
    # --viz_batch_level