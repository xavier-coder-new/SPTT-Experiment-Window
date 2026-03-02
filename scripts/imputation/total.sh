#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mask_rate=0.25

#################################################################################################
seed=2026

###################ETTh1###################
data_name=ETTh1
echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"
echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# ###################ETTh2###################
data_name=ETTh2
echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"
echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# ###################ETTm1###################
data_name=ETTm1
echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"
echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# ###################ETTm2###################
data_name=ETTm2
echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"
echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# ###################weather###################
data_name=weather
echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"
echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# ###################electricity###################
data_name=electricity
echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"
echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

#################################################################################################


# seed=2024

# ###################ETTh1###################
# data_name=ETTh1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTh2###################
# data_name=ETTh2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm1###################
# data_name=ETTm1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"
# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm2###################
# data_name=ETTm2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################weather###################
# data_name=weather
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################electricity###################
# data_name=electricity
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."


# #######################################################################################3


# seed=2026

# ###################ETTh1###################
# data_name=ETTh1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"


# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTh2###################
# data_name=ETTh2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm1###################
# data_name=ETTm1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm2###################
# data_name=ETTm2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################weather###################
# data_name=weather
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################electricity###################
# data_name=electricity
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Bptt/BpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"

# ./scripts/imputation/Bptt/BpttGRU.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."


######################################################################################

# seed=2026

# ###################ETTh1###################
# data_name=ETTh1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"


# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTh2###################
# data_name=ETTh2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm1###################
# data_name=ETTm1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm2###################
# data_name=ETTm2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################weather###################
# data_name=weather
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################electricity###################
# data_name=electricity
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."


# ##################################################################################


# seed=2027

# ###################ETTh1###################
# data_name=ETTh1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"


# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTh2###################
# data_name=ETTh2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm1###################
# data_name=ETTm1
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################ETTm2###################
# data_name=ETTm2
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################weather###################
# data_name=weather
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."

# # ###################electricity###################
# data_name=electricity
# echo "开始运行 $data_name ..., seed=$seed ..., mask_rate=$mask_rate ..."
# ./scripts/imputation/Sptt/SpttLSTM.sh "$data_name" "$seed" "$mask_rate"
# ./scripts/imputation/Sptt/SpttGRU.sh "$data_name" "$seed" "$mask_rate"

# echo "$data_name 运行完成!, seed=$seed ..., mask_rate=$mask_rate ..."