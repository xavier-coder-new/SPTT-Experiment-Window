#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 获取命令行参数
# name=$1
# 这样运行bash scripts_name.sh ETTh1, 这样ETTh1自动赋值给name变量了。
# 

model_name=SpttGRUNew
data_name=$1
seed=$2
echo "当前数据集: $data_name"
# M: 多变量预测多变量， MS: 多变量预测单变量， S: 单变量预测单变量
features=M
train_epochs=300
patience=10
time_block_num=$3
krank=$4
learning_rate=$5
# inverse_model=no_align
inverse_model=$6
echo "接收的参数 $data_name, $seed, $time_block_num, $krank, $learning_rate, $inverse_model"

if [[ "$data_name" = "ETTh1" || "$data_name" = "ETTh2" || "$data_name" = "ETTm1" || "$data_name" = "ETTm2" ]]; then
    data_dict=$data_name
    root_path=./dataset/ETT-small/
    feature_dim=7
    if [[ "$data_name" = "ETTh1" || "$data_name" = "ETTh2" ]]; then
        echo "使用ETTh数据集"
        batch_size=64
        seq_len=384
        pred_len=96
        label=384_96
    elif [[ "$data_name" = "ETTm1" || "$data_name" = "ETTm2" ]]; then
        echo "使用ETTm数据集"
        batch_size=256
        # seq_len=192
        # pred_len=96
        # label=192_96
        seq_len=480
        pred_len=192
        label=480_192
    fi
else
    data_dict=custom
    root_path="./dataset/$data_name/"
    if [[ "$data_name" = "traffic" ]]; then
        echo "使用traffic数据集"
        feature_dim=862
        batch_size=64
        seq_len=480
        pred_len=192
        label=480_192
    elif [[ "$data_name" = "electricity" ]]; then
        echo "使用electricity数据集"
        feature_dim=321
        batch_size=128
        seq_len=672
        pred_len=384
        label=672_384
    elif [[ "$data_name" = "weather" ]]; then
        echo "使用weather数据集"
        feature_dim=21
        batch_size=128
        seq_len=672
        pred_len=384
        label=672_384
    elif [[ "$data_name" = "illness" ]]; then
        echo "使用illness数据集"
        patience=30
        feature_dim=7
        batch_size=32
        seq_len=96
        pred_len=96
        label=96_96
    elif [[ "$data_name" = "exchange_rate" ]]; then
        echo "使用exchange_rate数据集"
        feature_dim=8
        batch_size=64
        seq_len=192
        pred_len=96
        label=192_96
    fi
fi

# batch_size 对于ETTm 64太小了。
# itr 实验次数
# type3 余弦退火，type2：阶段衰减
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
    --estimation_method Hut \
    --inverse_model $inverse_model

