#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

seed=2024

# ###################ETTh1###################
# data_name=ETTh1
# echo "开始运行 $data_name, seed=$seed..."
# ./scripts/long_term_forecast/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttLSTM.sh "$data_name" "$seed"


# ./scripts/long_term_forecast/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttGRU.sh "$data_name" "$seed"
# echo "$data_name 运行完成, seed=$seed!!!!!!"

# ####################ETTh2###################
data_name=electricity
echo "开始运行$data_name, seed=$seed..."
./scripts/long_term_forecast/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttLSTM.sh "$data_name" "$seed"


# ./scripts/long_term_forecast/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttGRU.sh "$data_name" "$seed"
echo "$data_name 运行完成, seed=$seed!!!!!!"


# # # ####################weather#######################
# data_name=weather
# echo "开始运行$data_name, seed=$seed..."
# ./scripts/long_term_forecast/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttLSTM.sh "$data_name" "$seed"


# ./scripts/long_term_forecast/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttGRU.sh "$data_name" "$seed"
# echo "$data_name 运行完成, seed=$seed!!!!!!"

# # ####################exchange_rate###################
# data_name=exchange_rate
# echo "开始运行$data_name, seed=$seed..."
# ./scripts/long_term_forecast/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttLSTM.sh "$data_name" "$seed"


# ./scripts/long_term_forecast/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/long_term_forecast/Bptt/BpttGRU.sh "$data_name" "$seed"
# echo "$data_name 运行完成, seed=$seed!!!!!!"

###################################################################################################
