#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TBptt_Accum
data_name=DelayedCopy
seed=$1
time_block_num=$2
krank=7
learning_rate=0.001
inverse_model=align
train_epochs=300
patience=25
batch_size=256
feature_dim=16

seq_len=$3
noise_std=$4
num_classes=$5

root_path="./dataset/delayed_copy_seq_${seq_len}_noise_${noise_std}_class_${num_classes}"
echo "root_path: $root_path"
label=$feature_dim'_'$num_classes'_'$seq_len

python -u run.py \
  --task_name sequence_delay \
  --is_training 1 \
  --model_id "$data_name'_'$label" \
  --model $model_name \
  --seed $seed \
  --data DelayedCopy \
  --data_name "$data_name" \
  --root_path "$root_path" \
  --krank $krank \
  --time_block_num $time_block_num \
  --feature_dim "$feature_dim" \
  --itr 1 \
  --train_epochs "$train_epochs" \
  --batch_size "$batch_size" \
  --patience "$patience" \
  --learning_rate $learning_rate \
  --lradj type3 \
  --tmax 10 \
  --use_gpu True \
  --gpu 0 \
  --gpu_type cuda \
  --hidden_dim 512 \
  --estimation_method None \
  --num_classes "$num_classes" \
  --des 'Exp' \
  --inverse_model $inverse_model \
  --use_chunk 1

