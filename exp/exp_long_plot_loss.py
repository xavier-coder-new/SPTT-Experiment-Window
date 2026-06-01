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

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        
    logger = logging.getLogger("long_term_forecast")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    print("time_block: ", time_block)
    
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
    
    original_emit = file_handler.emit
    def new_emit(record):
        result = original_emit(record)
        file_handler.flush()
        return result
    file_handler.emit = new_emit
    
    logger.addHandler(file_handler)

    return logger

class Exp_Long_Term_Forecast_Plot_Loss(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Plot_Loss, self).__init__(args)

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
            speed_iter_count = 0 
            train_loss = []
            total_time = 0.0
            min_time_batch = float('inf')
            max_time_batch = 0.0
            
            self.model.train()
            epoch_time = time.time()        
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                speed_iter_count += 1
                model_optim.zero_grad() 
                
                if self.is_first_train:
                    print(self.args.data_path , batch_x.shape , batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
                    # x shape [64, 384, 7] y shape [64, 144, 7] x_mark shape[64, 384, 4], y_mark shape [64, 144, 4]
                    self.is_first_train = False
                
                # the input time series data(features) used for prediction
                batch_x = batch_x.float().to(self.device)
                # the target time series data(values to be predicted)
                batch_y = batch_y.float().to(self.device)

                loss, sum_loss = self.model(batch_x, batch_y, criterion, model_optim)
                
                avg_loss = sum_loss / self.args.time_block_num
                
                train_loss.append(avg_loss)
                
                global_iter = epoch * train_steps + iter_count

                train_loss_records.append({
                    "epoch": epoch_count,
                    "batch": iter_count,
                    "global_iter": global_iter,
                    "train_loss": avg_loss,
                })
                

                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch_count, loss.item()))
                    speed = (time.time() - time_now) / speed_iter_count
                    left_time = speed * ((self.args.train_epochs - epoch_count) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    speed_iter_count = 0 
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

            train_loss = np.average(train_loss)
            
            vali_loss_av, vali_batch_losses = self.vali(vali_data, vali_loader, criterion)
            
            for v_batch_idx, v_loss in enumerate(vali_batch_losses, start=1):
                vali_loss_records.append({
                    "epoch": epoch_count,
                    "batch": v_batch_idx,
                    "global_iter": epoch * len(vali_loader) + v_batch_idx,
                    "vali_loss": v_loss,
                })
                
            epoch_loss_records.append({
                "epoch": epoch_count,
                "train_loss": float(train_loss),
                "vali_loss": float(vali_loss_av),
            })
            
            if self.args.lradj == 'type1':
                adjust_learning_rate(model_optim, epoch_count, self.args)
            else:
                scheduler.step()
                
            logger.info(
                f"Epoch: {epoch_count}, Steps: {train_steps} lr: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss_av:.7f} Total_time: {total_time:.5f} Min_time: {min_time_batch:.5f} Max_time: {max_time_batch:.5f}"
            )
            
            avg_time = total_time / train_steps
            
            logger.info(
                f"Epoch: {epoch_count}, Steps: {train_steps} avg_time_per_batch: {avg_time}"
            )
            
            if avg_time < min_time_epoch:
                min_time_epoch = avg_time
            if avg_time > max_time_epoch:
                max_time_epoch = avg_time
            
            avg_time_epoch += avg_time
            
            if vali_loss_av < best_vali_loss:
                best_vali_loss = vali_loss_av
                best_epoch = epoch + 1
            
            early_stopping(vali_loss_av, self.model, path)
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

        if best_epoch == 0:
            best_epoch = epoch_count

        train_loss_full_df = pd.DataFrame(train_loss_records)
        vali_loss_full_df = pd.DataFrame(vali_loss_records)
        epoch_loss_full_df = pd.DataFrame(epoch_loss_records)

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

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                loss = self.model(batch_x, batch_y, criterion)

                total_loss.append(loss.item())
                
        total_loss_avg = np.average(total_loss)
        self.model.train()
        return total_loss_avg, total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            load_path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), self.args.inverse_model, setting)
            self.model.load_state_dict(torch.load(os.path.join(load_path, 'checkpoint.pth')))

        preds = []
        trues = []

        folder_path = os.path.join('./test_results', self.args.task_name, self.args.model, 
                                   self.args.data_name, str(self.args.seed), self.args.inverse_model, str(self.timestamp_loss_csv))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                        
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                # pred: (batch_size, pred_len, feature_dim)
                pred = outputs
                true = batch_y

                # preds: (number, batch_size, pred_len, feature_dim)
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
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        logger.info(f"Test | mse:{mse:.4f}  mae:{mae:.4f}  rmse:{rmse:.4f}  mape:{mape:.4f}  mspe:{mspe:.4f}")
        
        return
