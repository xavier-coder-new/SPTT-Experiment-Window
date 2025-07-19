#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 获取命令行参数
# name=$1
# 这样运行bash scripts_name.sh ETTh1, 这样ETTh1自动赋值给name变量了。
# 

model_name=SpttLSTM_T
seed=$2
data_name=$1
# M: 多变量预测多变量， MS: 多变量预测单变量， S: 单变量预测单变量
features=M
train_epochs=300
pred_len=0

anomaly_ratio=$4

root_path=./dataset/"$data_name"/
batch_size=$5

# MSL 55, PSM 25, SWAP 25, SWD 38, SWAT 51

if [[ "$data_name" == "MSL" ]]; then
    # seq_len后面可加长一些。
    seq_len=$3
    feature_dim=55
elif [[ "$data_name" == "PSM" ]]; then
    seq_len=$3
    feature_dim=25
elif [[ "$data_name" == "SMAP" ]]; then
    seq_len=$3
    feature_dim=25
elif [[ "$data_name" == "SMD" ]]; then
    seq_len=$3
    feature_dim=38
elif [[ "$data_name" == "SWAT" ]]; then
    seq_len=$3
    feature_dim=51
fi

label=$seq_len'_'$pred_len

# batch_size 对于ETTm 64太小了。
# itr 实验次数
# type3 余弦退火，type2：阶段衰减
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
    --krank 10 \
    --time_block_num 4 \
    --feature_dim "$feature_dim" \
    --itr 1 \
    --train_epochs $train_epochs \
    --batch_size "$batch_size" \
    --patience 10 \
    --learning_rate 0.0001 \
    --loss MSE \
    --lradj type3 \
    --tmax 10 \
    --use_gpu True \
    --gpu 0 \
    --gpu_type cuda \
    --hidden_dim 512 \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --anomaly_ratio $anomaly_ratio \
    --estimation_method weihao

