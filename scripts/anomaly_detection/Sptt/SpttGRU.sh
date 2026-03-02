#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
model_name=SpttGRU
data_name=$1
seed=$2
time_block_num=$3
krank=$4
learning_rate=$5
inverse_model=$6
# M: 多变量预测多变量， MS: 多变量预测单变量， S: 单变量预测单变量
features=M
train_epochs=100
pred_len=0
patience=10

anomaly_ratio=1

root_path=./dataset/"$data_name"/


# MSL 55, PSM 25, SWAP 25, SWD 38, SWAT 51

if [[ "$data_name" == "MSL" ]]; then
    # seq_len后面可加长一些。
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

# seq_len=100

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
    --estimation_method weihao \
    --inverse_model $inverse_model \
    # --record \
    # --viz_batch_level

