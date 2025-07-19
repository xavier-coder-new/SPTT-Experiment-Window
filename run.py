import os
import sys
import time 
import torch
import argparse
import random
import numpy as np
from utils.print_args import print_args
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from global_param import global_param

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description="STITT for time series forecasting")
    
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options: [long_term_forecast, short_term_forecast, classification]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='normal_lstm',
                        help='model name, options: [BpttLSTM, BpttGRU, StittLSTM, StittGRU]')
    parser.add_argument('--seed', type=int, required=True, default=2025, help='random seed')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned], for m4 dataset')
    
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type, UEA for classification')
    parser.add_argument('--root_path', type=str, required=True, default='./dataset/ETT-small/',
                        help='root path of the dataset')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; \
                            M:multivariat predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, q:quarterly, a:annually]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # action='store_true', 开关型命令行参数
    # #当用户在命令行中添加 --inverse 参数时，它会将 args.inverse 设置为 True，而如果没有提供该参数，则 args.inverse 会默认是 False。
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    # model define
    parser.add_argument('--krank', type=int, default=10, help='SVD rank for STITT')
    parser.add_argument('--time_block_num', type=int, default=4, help='time block num for STITT')
    parser.add_argument('--data_name', type=str, required=True, default='ETTh1', help='dataset name')
    # c_out:输出维度，定义了模型最终的预测输出维度。表示模型在每个预测时间步需要输出多少个变量。
    # 直接决定了模型的预测结果的形式。
    parser.add_argument('--feature_dim', type=int, required=True, default=7, help='feature dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    
    # optimizer
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=6, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate \
                            type1:指数衰减, type2:阶段衰减, type3:余弦退火')
    parser.add_argument('--tmax', type=int, default=10)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)

    # RNN basic hyper params
    parser.add_argument('--hidden_dim', type=int, required=True, default=512, help='RNN hidden unit')    
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # label_len： 解码器起始长度，这个值通常用于Transformer类模型的解码器部分。
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--dropout_x', type=float, default=0.0, help='dropout for X')
    parser.add_argument('--dropout_d', type=float, default=0.0, help='dropout for D')
    parser.add_argument('--hutchinson_size', type=int, default=100, help='Hutchinson sample size')
    parser.add_argument('--random_size', type=int, default=100, help='random sample size')
    parser.add_argument('--estimation_method', type=str, default='Hut', help='estimation method: Hut, Random, Dropout, None')
    parser.add_argument('--trials', type=int, default=5, help='trials for Dropout estimation')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes for classification')
    
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly_detection task
    parser.add_argument('--anomaly_ratio', type=float, default=1, help='anomaly ratio')
    
    args = parser.parse_args()
    
    print("time_block_num: ", args.time_block_num)
    
    fix_seed = args.seed
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    
    global_param['trials'] = args.trials
    global_param['dropout_x'] = args.dropout_x
    global_param['dropout_d'] = args.dropout_d
    global_param['hutchinson_size'] = args.hutchinson_size
    global_param['random_size'] = args.random_size
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    print('Arg in experiment')
    # print_args(args)
    
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    else:
        raise NotImplementedError
    
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{0}_{1}_{2}_{3}_sd_{4}_ft{5}_sl{6}_ll{7}_pl{8}_Hut{9}_Rand{10}_Drop{11}{12}_{13}_{14}_{15}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data_name,
                args.seed,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.hutchinson_size,
                args.random_size,
                args.dropout_x,
                args.dropout_d,
                # args.d_ff, 全连接层的维度
                args.embed,
                args.des, ii)
            
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>start testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        # 这个地方用于加载已经训练好的模型参数
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{0}_{1}_{2}_{3}_sd_{4}_ft{5}_sl{6}_ll{7}_pl{8}_Hut{9}_Rand{10}_Drop{11}{12}_{13}_{14}_{15}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data_name,
            args.seed,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.hutchinson_size,
            args.random_size,
            args.dropout_x,
            args.dropout_d,
            # args.d_ff, 全连接层的维度
            args.embed,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()