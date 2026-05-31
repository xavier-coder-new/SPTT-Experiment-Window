from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, visual_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import logging
import torch.optim.lr_scheduler as lr_scheduler
from global_param import global_param
import csv
import pandas as pd
from icecream import ic
from utils.visualization_spectrum_drift import plot_spectrum_drift

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
if not os.path.exists(log_root):
    os.makedirs(log_root)

def setup_logger(model_name, dataset_name, seed, learning_rate, krank, time_block, 
                    batch_size, seq_len, pred_len, features, task_name, patience, epoch, inverse_model):
    log_project = os.path.join(log_root, f"{task_name}", f"{model_name}")
    if not os.path.exists(log_project):
        os.makedirs(log_project)
        
    log_dir = os.path.join(log_project, f'{dataset_name}', inverse_model, f'seed_{seed}', f'Pa_Ep:{patience}_{epoch}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建logger
    logger = logging.getLogger("classification")
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
    
    logger.addHandler(file_handler)

    return logger

class Exp_Classification_Plot_Loss(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification_Plot_Loss, self).__init__(args)

    # Exp_Basic初始化的时候就会建造模型
    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        # max(a, b)返回a,b中的最大值。
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        self.is_first_train = True
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # 对于batch_size较小的情况，RAdam可能更合适
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        # test_data, test_loader = self._get_data(flag='TEST')
        
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

        path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), self.args.inverse_model, setting)
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

        # ===============================
        # Loss records for clean exporting
        # ===============================
        
        train_loss_records = []
        vali_loss_records = []
        epoch_loss_records = []
        
        best_epoch = 0
        best_vali_loss = float("inf")
        stopped_epoch = None
        
        for epoch in range(0, self.args.train_epochs):
            epoch_count += 1
            iter_count = 0
            train_loss = []
            total_time = 0.0
            min_time_batch = float('inf')
            max_time_batch = 0.0
            
            self.model.train()
            epoch_time = time.time()
            
            preds_train = []
            trues_train = []

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.is_first_train:
                    # 对于EthanolConcentration torch.Size([64, 1751, 3]) torch.Size([64, 1751])
                    print(f"batch_x shape:{batch_x.shape}, padding_mask.shape:{padding_mask.shape}")
                    self.is_first_train = False
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                # outputs = self.model(batch_x)
                outputs, loss, sum_loss = self.model(batch_x, label, criterion, model_optim)
                
                avg_loss = sum_loss / self.args.time_block_num
                
                # loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(avg_loss)
                # train_loss_per_iter.append(avg_loss)
                
                global_iter = epoch * train_steps + iter_count

                train_loss_records.append({
                    "epoch": epoch + 1,
                    "batch": iter_count,
                    "global_iter": global_iter,
                    "train_loss": avg_loss,
                })
                
                preds_train.append(outputs.detach())
                trues_train.append(label)

                if (i + 1) % 100 == 0:
                    # \t是制表符，用于在打印时缩进，让输出更加整齐
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 计算每次迭代的平均耗时。
                    speed = (time.time() - time_now) / iter_count
                    # 估算剩余训练时间
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
                
                # 梯度裁剪
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
                logger.info(f"Epoch: {epoch + 1} Batch: {iter_count} Loss: {loss.item()} Time_batch: {time_for_batch:.5f}")
                
            train_loss = np.average(train_loss)
            # train_loss_per_iter.append(train_loss)
            
            preds_train = torch.cat(preds_train, 0)
            trues_train = torch.cat(trues_train, 0)
            probs_train = torch.nn.functional.softmax(preds_train, dim=1) 
            predictions_train = torch.argmax(probs_train, dim=1).cpu().numpy()
            trues_train = trues_train.flatten().cpu().numpy()
            
            accuracy_train = cal_accuracy(predictions_train, trues_train)
            
            vali_loss, val_accuracy, vali_batch_losses = self.vali(vali_data, vali_loader, criterion)
            
            for v_batch_idx, v_loss in enumerate(vali_batch_losses, start=1):
                vali_loss_records.append({
                    "epoch": epoch + 1,
                    "batch": v_batch_idx,
                    "global_iter": epoch * len(vali_loader) + v_batch_idx,
                    "vali_loss": v_loss,
                })
                
            epoch_loss_records.append({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "vali_loss": float(vali_loss),
                "train_acc": float(accuracy_train),
                "vali_acc": float(val_accuracy),
            })
            
            if self.args.lradj == 'type1':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                scheduler.step()
                
            logger.info(
                f"Epoch: {epoch + 1}, Steps: {train_steps} lr: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Total_time: {total_time:.5f} Min_time: {min_time_batch:.5f} Max_time: {max_time_batch:.5f} Train Acc: {accuracy_train:.3f} Vali Acc: {val_accuracy:.3f} cost time: {time.time() - epoch_time}"
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
            
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                best_epoch = epoch + 1
            
            # if self.args.data_name == 'SelfRegulationSCP1' or self.args.data_name == 'FaceDetection':
            # if self.args.data_name == 'SelfRegulationSCP1':
            # # early_stopping(vali_loss, self.model, path)
            #     early_stopping(-val_accuracy, self.model, path)
            # else:
            #     # HACK
            early_stopping(vali_loss, self.model, path)
                # early_stopping(-val_accuracy, self.model, path)
            # early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                
                avg_time_epoch /= epoch_count
                logger.info(f"Early stopping Epoch: {epoch + 1} Steps: {train_steps} Average_time_epoch {avg_time_epoch:.5f} Min_time_epoch: {min_time_epoch:.5f} Max_time_epoch: {max_time_epoch:.5f}")
                break

        # =====================================================
        # Export loss records after removing only the final
        # early-stopping tail.
        #
        # Note:
        # This does not remove intermediate non-improving epochs.
        # It only removes epochs after the final best epoch.
        # =====================================================

        if stopped_epoch is None:
            stopped_epoch = epoch_count

        # 如果训练过程中没有刷新 best_epoch，做一个保险处理
        if best_epoch == 0:
            best_epoch = epoch_count

        train_loss_full_df = pd.DataFrame(train_loss_records)
        vali_loss_full_df = pd.DataFrame(vali_loss_records)
        epoch_loss_full_df = pd.DataFrame(epoch_loss_records)

        # 标记哪些 epoch 用于最终绘图
        # 这里保留 epoch <= best_epoch
        # 因此，中间短暂没有 improvement 的 epoch 不会被删除
        # 只删除最后 best_epoch 之后的 early-stopping tail
        train_loss_full_df["used_for_plot"] = (
            train_loss_full_df["epoch"] <= best_epoch
        ).astype(int)

        vali_loss_full_df["used_for_plot"] = (
            vali_loss_full_df["epoch"] <= best_epoch
        ).astype(int)

        epoch_loss_full_df["used_for_plot"] = (
            epoch_loss_full_df["epoch"] <= best_epoch
        ).astype(int)

        train_loss_plot_df = train_loss_full_df[
            train_loss_full_df["used_for_plot"] == 1
        ].copy()

        vali_loss_plot_df = vali_loss_full_df[
            vali_loss_full_df["used_for_plot"] == 1
        ].copy()

        epoch_loss_plot_df = epoch_loss_full_df[
            epoch_loss_full_df["used_for_plot"] == 1
        ].copy()

        loss_export_dir = os.path.join(
            './loss_results',
            self.args.task_name,
            self.args.model,
            self.args.data_name,
            str(self.args.seed),
            self.args.inverse_model,
            f"Patience_{self.args.patience}_Epoch_{self.args.train_epochs}",
            f"Time_block_{self.args.time_block_num}_Krank_{self.args.krank}",
        )

        os.makedirs(loss_export_dir, exist_ok=True)

        # 完整 loss：包含最终 early-stopping tail，便于之后核查实验过程
        train_loss_full_df.to_csv(
            os.path.join(loss_export_dir, f"train_loss_full_seed_{self.args.seed}.csv"),
            index=False
        )

        vali_loss_full_df.to_csv(
            os.path.join(loss_export_dir, f"vali_loss_full_seed_{self.args.seed}.csv"),
            index=False
        )

        epoch_loss_full_df.to_csv(
            os.path.join(loss_export_dir, f"epoch_loss_full_seed_{self.args.seed}.csv"),
            index=False
        )

        # 绘图 loss：去掉最终 early-stopping tail
        train_loss_plot_df.to_csv(
            os.path.join(loss_export_dir, f"train_loss_plot_seed_{self.args.seed}.csv"),
            index=False
        )

        vali_loss_plot_df.to_csv(
            os.path.join(loss_export_dir, f"vali_loss_plot_seed_{self.args.seed}.csv"),
            index=False
        )

        epoch_loss_plot_df.to_csv(
            os.path.join(loss_export_dir, f"epoch_loss_plot_seed_{self.args.seed}.csv"),
            index=False
        )

        # 顺手生成单模型 loss 图
        visual_loss(
            train_loss_plot_df["train_loss"].tolist(),
            "Train Loss",
            os.path.join(loss_export_dir, "train_loss_plot.svg")
        )

        visual_loss(
            vali_loss_plot_df["vali_loss"].tolist(),
            "Validation Loss",
            os.path.join(loss_export_dir, "vali_loss_plot.svg")
        )

        logger.info(
            f"Loss exported. Best epoch: {best_epoch}, stopped epoch: {stopped_epoch}. "
            f"Only epochs after the final best epoch are excluded from plot files."
        )
        # 先在验证集上验证，然后在训练循环结束后，添加测试代码来评估模型在测试集上的性能。
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                outputs = self.model(batch_x, label, criterion)

                pred = outputs.detach().cpu()
                
                # # label.long().squeeze().cpu(): 确保label是整数，并去除多余维度，移动到CPU。
                # loss = criterion(pred, label.long().squeeze().cpu())
                processed_label = label.long().squeeze(-1).cpu()
                
                loss = criterion(pred, processed_label)
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)
        
        total_loss_avg = np.average(total_loss)

        # preds和trues是两个列表，分别存储多个batch的结果
        # torch.cat(preds, 0)沿着batch维度拼接，将所有batch的结果连接起来，得到完整的预测值和真实标签。
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # torch.argmax(probs, dim=1) 找到每个样本预测概率最高的类别索引。
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        # 展平 trues 并转换为 NumPy
        trues = trues.flatten().cpu().numpy()
        # 
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss_avg, accuracy, total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            # 对历史版本的时间序列预测的加载需要修改下路径
            # load_path = os.path.join('./checkpoints/', long_term_forecast, setting)
            load_path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), self.args.inverse_model, setting)
            self.model.load_state_dict(torch.load(os.path.join(load_path, 'checkpoint.pth')))

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        
        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())    
        # folder_path = os.path.join('./test_results', self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), str(timestamp))        
        
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # outputs = self.model(batch_x, padding_mask, None, None)
                outputs = self.model(batch_x)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        print('accuracy:{}'.format(accuracy))

        logger.info(f"Test | Accuracy: {accuracy}")
        
        return
