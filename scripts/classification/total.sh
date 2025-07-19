#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

seed=2018
####################EthanolConcentration###########################
data_name=EthanolConcentration
echo "开始运行 $data_name, seed=$seed..."
# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

#####################FaceDetection##################################
data_name=FaceDetection
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

#####################Handwriting##################################
data_name=Handwriting
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################Heartbeat#############################
data_name=Heartbeat
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################JapaneseVowels####################################
data_name=JapaneseVowels
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################PEMS-SF####################################
data_name=PEMS-SF
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"
echo "$data_name 运行完成!!!"


#####################SelfRegulationSCP1####################################
data_name=SelfRegulationSCP1
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################SelfRegulationSCP2####################################
data_name=SelfRegulationSCP2
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################SpokenArabicDigits####################################
data_name=SpokenArabicDigits
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

#################################################################################

seed=2024
####################EthanolConcentration###########################
data_name=EthanolConcentration
echo "开始运行 $data_name, seed=$seed..."


# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

#####################FaceDetection##################################
data_name=FaceDetection
echo "开始运行 $data_name, seed=$seed..."
# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

#####################Handwriting##################################
data_name=Handwriting
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################Heartbeat#############################
data_name=Heartbeat
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################JapaneseVowels####################################
data_name=JapaneseVowels
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################PEMS-SF####################################
data_name=PEMS-SF
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"
echo "$data_name 运行完成!!!"


#####################SelfRegulationSCP1####################################
data_name=SelfRegulationSCP1
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################SelfRegulationSCP2####################################
data_name=SelfRegulationSCP2
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################SpokenArabicDigits####################################
data_name=SpokenArabicDigits
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


########################################################


seed=2026
####################EthanolConcentration###########################
data_name=EthanolConcentration
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

#####################FaceDetection##################################
data_name=FaceDetection
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

#####################Handwriting##################################
data_name=Handwriting
echo "开始运行 $data_name, seed=$seed..."
# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################Heartbeat#############################
data_name=Heartbeat
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################JapaneseVowels####################################
data_name=JapaneseVowels
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################PEMS-SF####################################
data_name=PEMS-SF
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"
echo "$data_name 运行完成!!!"


#####################SelfRegulationSCP1####################################
data_name=SelfRegulationSCP1
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################SelfRegulationSCP2####################################
data_name=SelfRegulationSCP2
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"


#####################SpokenArabicDigits####################################
data_name=SpokenArabicDigits
echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

echo "$data_name 运行完成!!!"

####################################################################################


# seed=2026
# ####################EthanolConcentration###########################
# data_name=EthanolConcentration
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"

# #####################FaceDetection##################################
# data_name=FaceDetection
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"

# #####################Handwriting##################################
# data_name=Handwriting
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################Heartbeat#############################
# data_name=Heartbeat
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################JapaneseVowels####################################
# data_name=JapaneseVowels
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################PEMS-SF####################################
# data_name=PEMS-SF
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"
# echo "$data_name 运行完成!!!"


# #####################SelfRegulationSCP1####################################
# data_name=SelfRegulationSCP1
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################SelfRegulationSCP2####################################
# data_name=SelfRegulationSCP2
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################SpokenArabicDigits####################################
# data_name=SpokenArabicDigits
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed"



# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"

#########################################################################################

# seed=2027
# ####################EthanolConcentration###########################
# data_name=EthanolConcentration
# echo "开始运行 $data_name, seed=$seed..."


# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"

# #####################FaceDetection##################################
# data_name=FaceDetection
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"

# #####################Handwriting##################################
# data_name=Handwriting
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################Heartbeat#############################
# data_name=Heartbeat
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################JapaneseVowels####################################
# data_name=JapaneseVowels
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################PEMS-SF####################################
# data_name=PEMS-SF
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"
# echo "$data_name 运行完成!!!"


# #####################SelfRegulationSCP1####################################
# data_name=SelfRegulationSCP1
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################SelfRegulationSCP2####################################
# data_name=SelfRegulationSCP2
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"


# #####################SpokenArabicDigits####################################
# data_name=SpokenArabicDigits
# echo "开始运行 $data_name, seed=$seed..."

# ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed"
# ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed"

# echo "$data_name 运行完成!!!"
