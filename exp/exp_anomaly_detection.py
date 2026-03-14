from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, visual_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import csv
import logging
import torch.optim.lr_scheduler as lr_scheduler
from global_param import global_param
import pandas as pd

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
if not os.path.exists(log_root):
    os.makedirs(log_root)

def setup_logger(model_name, dataset_name, seed, learning_rate, krank, time_block, 
                    batch_size, seq_len, pred_len, features, task_name, anomaly_ratio,
                    patience, epoch, inverse_model):
    log_project = os.path.join(log_root, f"{task_name}", f"{model_name}")
    if not os.path.exists(log_project):
        os.makedirs(log_project)
        
    log_dir = os.path.join(log_project, f'{dataset_name}', inverse_model, f'seed_{seed}', f'anomaly_ratio_{anomaly_ratio}', f'Pa_Ep:{patience}_{epoch}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    
    # 创建logger
    logger = logging.getLogger("anomaly_ratio")
    logger.setLevel(logging.INFO)
    
    # 清除之前的处理器，防止重复
    if logger.handlers:
        logger.handlers.clear()
    
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

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        self.is_first_train = True
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
            self.args.anomaly_ratio,
            self.args.patience,
            self.args.train_epochs,
            self.args.inverse_model,
        )

        path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), str(self.args.anomaly_ratio), self.args.inverse_model, setting)
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
        train_loss_per_iter = []
        vali_loss_per_iter = []
        
        timestamp_loss_csv = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  
        train_loss_csv_dir = os.path.join(project_root, "loss_csv", self.args.task_name, self.args.model, self.args.data_name, 
                                      str(self.args.anomaly_ratio), str(self.args.seed), self.args.inverse_model, 'train')
        train_loss_csv_file = os.path.join(train_loss_csv_dir, f"train_loss_{timestamp_loss_csv}.csv")

        os.makedirs(train_loss_csv_dir,exist_ok=True)
        
        for epoch in range(self.args.train_epochs):
            epoch_count += 1
            iter_count = 0
            train_loss = []
            total_time = 0.0
            min_time_batch = float('inf')
            max_time_batch = 0.0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                if self.is_first_train:
                    print("batch_x shape: ", batch_x.shape)
                    self.is_first_train = False

                batch_x = batch_x.float().to(self.device)

                loss, sum_loss = self.model(batch_x, batch_y, criterion, model_optim)

                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, :, f_dim:]
                
                avg_loss = sum_loss / self.args.time_block_num
                
                # loss = criterion(outputs, batch_x)
                train_loss.append(avg_loss)
                # train_loss_per_iter.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                
                time_for_batch = global_param['time_for_batch']
                if time_for_batch > max_time_batch:
                    max_time_batch = time_for_batch
                if time_for_batch < min_time_batch:
                    min_time_batch = time_for_batch
                total_time += time_for_batch
                
                model_optim.step()
                
                logger.info(f"Epoch: {epoch + 1} Batch: {iter_count} Loss: {loss.item()} Time_batch: {time_for_batch:.5f}")


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
            
            # 写完之后再求均值清除
            train_loss = np.average(train_loss)
            
            # 此处是存均值
            train_loss_per_iter.append(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, timestamp_loss_csv)
            vali_loss_per_iter.append(vali_loss)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))

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
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())   
                folder_path_loss = os.path.join('./loss_results', self.args.task_name, self.args.model, self.args.data_name, str(self.args.anomaly_ratio), str(self.args.seed), self.args.inverse_model, 'train', str(timestamp))
                if not os.path.exists(folder_path_loss):
                    os.makedirs(folder_path_loss)
                
                visual_loss(train_loss_per_iter, 'Train Loss', os.path.join(folder_path_loss, 'train_loss.pdf'))
                
                with open(os.path.join(folder_path_loss, 'train_loss.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Train Loss'])  # 可选：写个列名
                    for loss in train_loss_per_iter:
                        writer.writerow([loss])
                
                folder_path_loss = os.path.join('./loss_results', self.args.task_name, self.args.model, self.args.data_name, str(self.args.anomaly_ratio), str(self.args.seed), self.args.inverse_model, 'vali', str(timestamp))
                if not os.path.exists(folder_path_loss):
                    os.makedirs(folder_path_loss)
                
                visual_loss(vali_loss_per_iter, 'Vali Loss', os.path.join(folder_path_loss, 'vali_loss.pdf'))
                
                with open(os.path.join(folder_path_loss, 'vali_loss.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Vali Loss'])  # 可选：写个列名
                    for loss in vali_loss_per_iter:
                        writer.writerow([loss])
                
                avg_time_epoch /= epoch_count
                train_loss_per_iter = []
                vali_loss_per_iter = []
                logger.info(f"Early stopping Epoch: {epoch + 1} Steps: {train_steps} Average_time_epoch {avg_time_epoch:.5f} Min_time_epoch: {min_time_epoch:.5f} Max_time_epoch: {max_time_epoch:.5f}")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def vali(self, vali_data, vali_loader, criterion, timestamp_loss_csv):
        total_loss = []
        self.model.eval()
        
        vali_loss_csv_dir = os.path.join(project_root, "loss_csv", self.args.task_name, self.args.model, 
                                     self.args.data_name, str(self.args.anomaly_ratio), str(self.args.seed), self.args.inverse_model, 'vali')
        vali_loss_csv_file = os.path.join(vali_loss_csv_dir, f"vali_loss_{timestamp_loss_csv}.csv")
        
        os.makedirs(vali_loss_csv_dir,exist_ok=True)
        
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                loss = self.model(batch_x, criterion=criterion)
                
                # outputs = self.model(batch_x)

                # f_dim = -1 if self.args.features == 'MS' else 0
                
                # outputs = outputs[:, :, f_dim:]
                
                # pred = outputs.detach().cpu()
                # true = batch_x.detach().cpu()

                # loss = criterion(pred, true)
                
                total_loss.append(loss.item())
        
        vali_loss_data = pd.DataFrame({
            'vali_loss': total_loss
        })
        
        if os.path.exists(vali_loss_csv_file):
            vali_loss_data.to_csv(vali_loss_csv_file, mode='a', header=False, index=False)
        else:
            vali_loss_data.to_csv(vali_loss_csv_file, mode='w', header=True, index=False)
        
        total_loss_avg = np.average(total_loss)
        self.model.train()
        return total_loss_avg

    def test(self, setting, test=0):
        # 在异常检测任务中，train_loader 在测试阶段使用，是为了统计正常样本的分布特征，用来估计"异常能量阈值"
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            # 构建与训练时一致的模型路径
            model_path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, 
                                     self.args.data_name, str(self.args.seed), str(self.args.anomaly_ratio), 
                                     self.args.inverse_model, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(model_path))

        attens_energy = []

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # 基于重构误差的异常检测流程。
        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion 对特征维度求平均，变成每个样本一个分数。
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                # 将这个batch的所有样本分数加入列表。
                attens_energy.append(score)

        # 将所有batch的分数拼接起来，变成一个一维数组。
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        print("attens_energy shape: ", attens_energy.shape)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            
            test_labels.append(batch_y)
        
        # np.concatenate 是 NumPy 提供的一个函数，用于将多个数组沿指定的轴（axis）进行连接。
        # .reshape(-1) 是 NumPy 中用于调整数组形状的方法。
        # 参数 -1 表示自动计算该维度的大小，使得数组的所有元素保持不变。
        # 结果是一个一维数组（即展平后的数组）。
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        
        # 把训练集 + 测试集的重构误差放在一起看，估计一个合理的分位点作为阈值
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        
        # self.args.anomaly_ratio是假设的异常数据占比。
        # np.percentile 是 NumPy 提供的一个函数，用于计算数组中某个百分位数的值。
        # 百分位数（Percentile）是统计学中的一个概念，表示在一组数据中，有多少比例的数据小于或等于该值。
        # np.percentile 使用线性插值来计算百分位数的具体值
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        # 将每个测试样本的重构误差与阈值比较， 大于阈值 ➝ 认为是异常（记为 1），否则为正常（记为 0）。
        pred = (test_energy > threshold).astype(int)
        
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        
        # gt: ground truth 表示真实标签
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        
        logger.info("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        
        # f = open("result_anomaly_detection.txt", 'a')
        # f.write(setting + "  \n")
        # f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #     accuracy, precision,
        #     recall, f_score))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        return
