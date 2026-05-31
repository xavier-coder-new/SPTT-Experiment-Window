#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=BpttSRU
data_name=$1
seed=$2
time_block_num=$3
krank=$4
learning_rate=$5
inverse_model=$6

train_epochs=300
patience=25
root_path=./dataset/"$data_name"/
# 特征维度，就是通道数，几通道代表几特征，每个时间步的输入维度。
# feature_dim=3
# 这个pred_len对于分类任务而言代表的类别数量。
# num_classes=4
if [[ "$data_name" == "SelfRegulationSCP1" ]]; then
  train_epochs=300
  patience=30
  batch_size=16
  feature_dim=6
  num_classes=2
  seq_len=896
elif [[ "$data_name" == "EthanolConcentration" ]]; then
  train_epochs=300
  patience=30
  batch_size=16
  feature_dim=3
  num_classes=4
  seq_len=1751
elif [[ "$data_name" == "SelfRegulationSCP2" ]]; then
  train_epochs=300
  patience=30
  batch_size=16
  feature_dim=7
  num_classes=2
  seq_len=1152
elif [[ "$data_name" == "FaceDetection" ]]; then
  train_epochs=300
  patience=30
  batch_size=32
  feature_dim=144
  num_classes=2
  seq_len=62
elif [[ "$data_name" == "Handwriting" ]]; then
  train_epochs=300
  patience=30
  batch_size=16
  feature_dim=3
  num_classes=26
  seq_len=152
elif [[ "$data_name" == "Heartbeat" ]]; then
  train_epochs=300
  patience=30
  batch_size=16
  feature_dim=61
  num_classes=2
  seq_len=405
elif [[ "$data_name" == "JapaneseVowels" ]]; then
  # 太垃圾了，不用。
  # train_epochs=300
  # patience=30
  train_epochs=800
  patience=100
  batch_size=16
  feature_dim=12
  num_classes=9
  seq_len=29
elif [[ "$data_name" == "PEMS-SF" ]]; then
  # train_epochs=300
  # patience=30
  train_epochs=800
  patience=100
  batch_size=16
  feature_dim=963
  num_classes=7
  seq_len=144
elif [[ "$data_name" == "SpokenArabicDigits" ]]; then
  train_epochs=1200
  patience=300
  batch_size=64
  feature_dim=13
  num_classes=10
  seq_len=93
elif [[ "$data_name" == "UWaveGestureLibrary" ]]; then
  # 太垃圾了，不用
  batch_size=16
  feature_dim=3
  num_classes=8
  seq_len=315
fi
label=$feature_dim'_'$num_classes'_'$seq_len

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --model_id "$data_name'_'$label" \
  --model $model_name \
  --seed $seed \
  --data UEA \
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
  --num_class "$num_classes" \
  --des 'Exp' \
  --inverse_model $inverse_model

