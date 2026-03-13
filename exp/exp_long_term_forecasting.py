from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_loss
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import torch.optim.lr_scheduler as lr_scheduler
from global_param import global_param
import logging
import csv
import pandas as pd
from icecream import ic
from utils.visualization_spectrum_drift import plot_spectrum_drift

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
输入: [batch_size, seq_len, feature_dim]
输出：[batch_size, pred_len, num_targets]
数据每次一个个滑动,构建输入序列。

"""

log_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
if not os.path.exists(log_root):
    os.makedirs(log_root)


def setup_logger(model_name, dataset_name, seed, learning_rate, krank, time_block, 
                    batch_size, seq_len, pred_len, features, task_name,
                    patience, epoch, inverse_model):
    log_project = os.path.join(log_root, f"{task_name}", f"{model_name}")
    if not os.path.exists(log_project):
        os.makedirs(log_project)
        
    log_dir = os.path.join(log_project, f'{dataset_name}', f'seed_{seed}', f'Pa_Ep:{patience}_{epoch}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    
    # 创建logger
    logger = logging.getLogger("long_term_forecast")
    logger.setLevel(logging.INFO)
    
    # 清除之前的处理器，防止重复
    if logger.handlers:
        logger.handlers.clear()
    
    print("time_block: ", time_block)
    
    # 创建文件处理器，并将日志文件保存到指定目录下
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    file_handler = logging.FileHandler(
        os.path.join(
            log_dir, 
            f"{features}_{learning_rate}_{batch_size}_{seq_len}_{pred_len}_{krank}_{time_block}_{timestamp}.log"
        ),
        mode='a'
    )
        
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 配置自动刷新
    original_emit = file_handler.emit
    def new_emit(record):
        result = original_emit(record)
        file_handler.flush()
        return result
    file_handler.emit = new_emit
    
    logger.addHandler(file_handler)

    return logger

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        self.is_first_train = True
        self.is_first_vali = True
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        timestamp_loss_csv = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.timestamp_loss_csv = timestamp_loss_csv
        
        global logger
        logger = setup_logger(
            self.args.model,
            self.args.data_name,
            self.args.seed,
            self.args.learning_rate,
            self.args.krank,
            self.args.time_block_num,
            self.args.batch_size,
            self.args.seq_len,
            self.args.pred_len,
            self.args.features,
            self.args.task_name,
            self.args.patience,
            self.args.train_epochs,
            self.args.inverse_model,
        )
        # log_file = os.path.join(self.args.checkpoints, setting, "train_log.txt")
        # with open(log_file, "w") as f:
        #     f.write("Epoch, Batch, Loss, Time_batch, Min_time, Max_time, LR\n")  # 写入标题

        path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("batch_num", train_steps)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.lradj == 'type2':
            scheduler = lr_scheduler.StepLR(model_optim, step_size=self.args.tmax, gamma=0.9)
        elif self.args.lradj == 'type3':
            scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs, eta_min=1e-6)

        min_time_epoch = float('inf')
        max_time_epoch = 0.0
        avg_time_epoch = 0.0
        epoch_count = 0
        
        train_loss_csv_dir = os.path.join('./loss_results', self.args.task_name, self.args.model, self.args.data_name, 
                                        str(self.args.seed), self.args.inverse_model, 'train',
                                        f"Patience_{self.args.patience}_Epoch_{self.args.train_epochs}", 
                                        f"Time_block_{self.args.time_block_num}_Krank_{self.args.krank}", 
                                        str(timestamp_loss_csv))
        train_loss_csv_file = os.path.join(train_loss_csv_dir, f"train_loss_{timestamp_loss_csv}.csv")
        os.makedirs(train_loss_csv_dir, exist_ok=True)
        
        vali_loss_csv_dir = os.path.join('./loss_results', self.args.task_name, self.args.model, self.args.data_name, 
                                         str(self.args.seed), self.args.inverse_model, 'vali',
                                        f"Patience_{self.args.patience}_Epoch_{self.args.train_epochs}", 
                                        f"Time_block_{self.args.time_block_num}_Krank_{self.args.krank}", 
                                        str(timestamp_loss_csv))
        vali_loss_csv_file = os.path.join(vali_loss_csv_dir, f"vali_loss_{timestamp_loss_csv}.csv")
        
        os.makedirs(vali_loss_csv_dir,exist_ok=True)
        
        for epoch in range(0, self.args.train_epochs):
            epoch_count += 1
            iter_count = 0
            train_loss = []
            total_time = 0.0
            min_time_batch = float('inf')
            max_time_batch = 0.0
            
            self.model.train()
            epoch_time = time.time()
            # 384_96_batch_num: 128            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # i其实就是batch数
                iter_count += 1
                model_optim.zero_grad() 
                
                if self.is_first_train:
                    print(self.args.data_path , batch_x.shape , batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                    # x shape [64, 384, 7] y shape [64, 144, 7] x_mark shape[64, 384, 4], y_mark shape [64, 144, 4]
                    self.is_first_train = False
                
                # the input time series data(features) used for prediction
                batch_x = batch_x.float().to(self.device)
                # the target time series data(values to be predicted)
                batch_y = batch_y.float().to(self.device)
                # Temporal features/markers corresponding to input data
                # 对应输入数据的时间特征或标记。这通常包含时间信息，如月份、星期几、小时等事件编码，帮助模型理解时间模式。
                # bacth_x_mark = batch_x_mark.float().to(self.device)
                # Temporal features/markers corresponding to target data
                # 对应于目标数据的时间特征或标记，与batch_x_mark类似，但是与预测目标时间点对应。
                # batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                # 这个loss为最后一个时间块的loss
                loss, sum_loss = self.model(batch_x, batch_y, criterion, model_optim)
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                # loss = criterion(outputs, batch_y)
                
                avg_loss = sum_loss / self.args.time_block_num
                
                train_loss.append(avg_loss)
                
                # 如果全连接层不对应修改的话，则使用下面程序
                # outputs = self.model(batch_x)
                # f_dim = -1 if self.args.features == 'MS' else 0
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y.squeeze(-1)
                # loss = criterion(outputs, batch_y)
                # train_loss.append(loss.item())

                if (i + 1) % 50 == 0:
                    # \t是制表符，用于在打印时缩进，让输出更加整齐。
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 计算每次迭代的平均耗时。
                    speed = (time.time() - time_now) / iter_count
                    # 估算剩余训练时间
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    # train_loss_avg50 = np.average(train_loss)
                    # train_loss_per_iter.append(train_loss_avg50) 

                loss.backward()
                
                if self.args.viz_spectrum_drift and self.args.model in {'BpttLSTM', 'BpttGRU'}:
                    grad_w_hh = None
                    for name, param in self.model.named_parameters():
                        if 'w_hh' in name and param.grad is not None:
                            grad_w_hh = param.grad.detach().float()
                            break
                    assert grad_w_hh is not None, "grad_w_hh is None"
                    if grad_w_hh is not None:    
                        if 'sigma_list' not in global_param:
                            ic("model name: ", self.args.model, "allow to record spectrum drift")
                            global_param['sigma_list'] = []
                        _, Sigma, _ = torch.linalg.svd(grad_w_hh, full_matrices=False)
                        truncated_sigma = Sigma[:self.args.krank]
                        global_param['sigma_list'].append(truncated_sigma.cpu().numpy())
                
                time_for_batch = global_param['time_for_batch']
                if time_for_batch > max_time_batch:
                    max_time_batch = time_for_batch
                if time_for_batch < min_time_batch:
                    min_time_batch = time_for_batch
                total_time += time_for_batch
                
                model_optim.step()
                
                logger.info(f"Epoch: {epoch + 1} Batch: {iter_count} Loss: {loss.item()} Time_batch: {time_for_batch:.5f}")

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            
            train_loss_data = pd.DataFrame({
                'train_loss': train_loss
            })
            
            # 追加写入csv文件
            if os.path.exists(train_loss_csv_file):
                # a:表示以追加模式打开文件，如果文件存在，新的数据会被追加到文件末尾。
                train_loss_data.to_csv(train_loss_csv_file, mode='a', header=False, index=False)
            else:
                # w:表示以写入模式打开文件，如果文件不存在，则会创建文件，如果文件存在，则会覆盖原有文件。
                train_loss_data.to_csv(train_loss_csv_file, mode='w', header=True, index=False)
            
            # NOTE: 求平均损失，求平均损失可以减少batch间的波动，稳定损失评估
            # 便于判断训练趋势，优化超参数，有助于绘制损失曲线，判断收敛情况。
            # 便于和验证集损失对比，检查是否过拟合。
            train_loss = np.average(train_loss)
            # train_loss_per_iter.append(train_loss)
            
            vali_loss = self.vali(vali_data, vali_loader, criterion, vali_loss_csv_file)

            # test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss))

            # 这里使用magic用法，直接调用EarlyStopping对象的__call__方法，传入验证集损失和模型、路径作为参数。
            
            if self.args.lradj == 'type1':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                scheduler.step()
                
            logger.info(
                f"Epoch: {epoch + 1}, Steps: {train_steps} lr: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Total_time: {total_time:.5f} Min_time: {min_time_batch:.5f} Max_time: {max_time_batch:.5f}"
            )
            
            avg_time = total_time / train_steps
            
            logger.info(
                f"Epoch: {epoch + 1}, Steps: {train_steps} avg_time_per_batch: {avg_time}"
            )
            
            if avg_time < min_time_epoch:
                min_time_epoch = avg_time
            if avg_time > max_time_epoch:
                max_time_epoch = avg_time
            
            avg_time_epoch += avg_time
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                train_loss_df = pd.read_csv(train_loss_csv_file)
                visual_loss(train_loss_df['train_loss'].tolist(), 'Train Loss', os.path.join(train_loss_csv_dir, 'train_loss.svg'))
                
                vali_loss_df = pd.read_csv(vali_loss_csv_file)
                visual_loss(vali_loss_df['vali_loss'].tolist(), 'Validation Loss', os.path.join(vali_loss_csv_dir, 'vali_loss.svg'))
                
                if self.args.viz_spectrum_drift:
                    ic("visualize spectrum drift")
                    viz_spectrum_drift_path = os.path.join('./viz_spectrum_drift', self.args.task_name, self.args.model, 
                                                           self.args.data_name, self.args.inverse_model, str(self.args.seed), 
                                                           f"Patience_{self.args.patience}_Epoch_{self.args.train_epochs}", 
                                                           f"Time_block_{self.args.time_block_num}_Krank_{self.args.krank}", 
                                                           str(timestamp_loss_csv))
                    if not os.path.exists(viz_spectrum_drift_path):
                        os.makedirs(viz_spectrum_drift_path)
                    plot_spectrum_drift(viz_spectrum_drift_path, global_param['sigma_list'], train_steps, self.args.patience, self.args.model)
                
                avg_time_epoch /= epoch_count
                logger.info(f"Early stopping Epoch: {epoch + 1} Steps: {train_steps} Average_time_epoch {avg_time_epoch:.5f} Min_time_epoch: {min_time_epoch:.5f} Max_time_epoch: {max_time_epoch:.5f}")
                break

        # 先在验证集上验证，然后在训练循环结束后，添加测试代码来评估模型在测试集上的性能。
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion, vali_loss_csv_file):
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                loss = self.model(batch_x, batch_y, criterion)
                
                # if self.is_first_vali:
                #     # outputs shape: [64, 96, 7]
                #     # batch_y shape: [64, 144, 7]
                #     print(f"outputs shape: {outputs.shape}, {outputs} \n")
                #     print(f"batch_y shape: {batch_y.shape}, {batch_y} \n")
                #     self.is_first_vali = False
                
                # f_dim = -1 if self.args.features == 'MS' else 0
                
                # # : batch维度不变，-self.args.pred_len: 取时间序列的最后pred_len个时间步。f_dim: 取特征维度全部或者最后一个特征。
                # # 所使用的数据集，最后一个特征维度是目标预测值。
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # # outputs是神经网络的计算图的一部分，它的梯度会被PyTorch跟踪。
                # # detach()会阻断梯度传播，让outputs变成一个不需要计算梯度的张量。
                # # 在验证阶段使用detach()可以避免不必要的计算和内存占用
                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()

                # loss = criterion(pred, true)

                total_loss.append(loss.item())
                
                # if (i + 1) % 50 == 0:
                #     vali_loss_av50 = np.average(total_loss)
                #     vali_loss_av.append(vali_loss_av50)
                
        vali_loss_data = pd.DataFrame({
            'vali_loss': total_loss
        })
        
        if os.path.exists(vali_loss_csv_file):
            vali_loss_data.to_csv(vali_loss_csv_file, mode='a', header=False, index=False)
        else:
            vali_loss_data.to_csv(vali_loss_csv_file, mode='w', header=True, index=False)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        # 用于只测试的时候
        if test:
            print('loading model')
            # path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), setting)
            load_path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), self.args.inverse_model, setting)
            self.model.load_state_dict(torch.load(os.path.join(load_path, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        folder_path = os.path.join('./test_results', self.args.task_name, self.args.model, 
                                   self.args.data_name, str(self.args.seed), self.args.inverse_model, str(self.timestamp_loss_csv))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # NOTE:  这里可能会有点歧义，小问题。
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # 如果数据test_data在预处理时被缩放，说明batch_y和outputs经过了标准化或归一化。
                # 如果self.args.inverse为True，说明我们需要对预测结果和真实值进行逆变换，将它们恢复到原始尺度。
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    # shape返回的是一个元组，表示数组在各个维度上的大小。-1取最后一个数值，表示特征维度的大小。
                    # 如果outputs.shape[-1] != batch_y.shape[-1]，说明输出维度和目标维度不一致，需要进行重复扩展。
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        # np.tile() 重复瓷砖操作，假设，np.tile(array, [1, 1, 3])
                        # 表示不复制batch维度，不复制时间步长维度，在特征维度复制三次，使得维度匹配。
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                        
                    # outputs.reshape(shape[0] * shape[1], -1) 将outputs变成二维矩阵(batch_size * seq_len, feature_dim)
                    # 然后再逆变换，将标准化/归一化的数据恢复到原始尺度。
                    # 逆变换后，数据仍然是二维(batch_size * seq_len, feature_dim)，需要再reshape成三维(batch_size, seq_len, feature_dim)。
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                # NOTE：如果outputs.shape[-1] == batch_y.shape[-1]，这两行代码是多余的。
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                # pred: (batch_size, pred_len, feature_dim)
                pred = outputs
                true = batch_y

                # preds: (number, batch_size, pred_len, feature_dim) 这些应该是画图，为了可视化。
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # preds: (number * batch_size, pred_len, feature_dim)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        
        # reshape(-1) 表示只对第一个维度进行调整。
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save ./ 表示当前工作目录
        # folder_path = os.path.join('./results/', self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), str(timestamp))
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        # # 如果没有，会自动创建，如果没有显示指定路径，那么文件会保存在当前工作目录，
        # # 也就是说，它会保存在你运行Python程序时所在的目录
        # # 在python中当前工作目录是指的启动脚本的地方。
        
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        logger.info(f"Test | mse:{mse:.4f}  mae:{mae:.4f}  rmse:{rmse:.4f}  mape:{mape:.4f}  mspe:{mspe:.4f}")
        
        # .npy文件时Numpy专用的二进制文件格式，用于高效存储多维数组。
        # np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        # np.save(os.path.join(folder_path, 'pred.npy'), preds)
        # np.save(os.path.join(folder_path, 'true.npy'), trues)

        return
