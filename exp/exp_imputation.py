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
import logging
import torch.optim.lr_scheduler as lr_scheduler
from global_param import global_param
import csv
import pandas as pd
from icecream import ic

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
if not os.path.exists(log_root):
    os.makedirs(log_root)

def setup_logger(model_name, dataset_name, seed, learning_rate, krank, time_block, 
                    batch_size, seq_len, pred_len, features, task_name, mask_rate,
                    patience, epoch, inverse_model):
    log_project = os.path.join(log_root, f"{task_name}", f"{model_name}")
    if not os.path.exists(log_project):
        os.makedirs(log_project)
        
    log_dir = os.path.join(log_project, f'{dataset_name}', f'seed_{seed}', f'maks_rate_{mask_rate}', f'Pa_Ep:{patience}_{epoch}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger("imputation")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
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

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

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
            self.args.mask_rate,
            self.args.patience,
            self.args.train_epochs,
            self.args.inverse_model,
        )
        
        path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, self.args.data_name, str(self.args.seed), self.args.inverse_model, str(self.args.mask_rate), setting)
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                if self.is_first_train:
                    print(self.args.data_path, batch_x.shape)
                    self.is_first_train = False

                batch_x = batch_x.float().to(self.device)

                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                
                inp = batch_x.masked_fill(mask == 0, 0)

                loss, sum_loss = self.model(batch_x, None, criterion, model_optim, inp=inp, mask=mask)
                
                avg_loss = sum_loss / self.args.time_block_num

                train_loss.append(avg_loss)


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
            
            train_loss_data = pd.DataFrame({
                'train_loss': train_loss
            })

            if os.path.exists(train_loss_csv_file):
                train_loss_data.to_csv(train_loss_csv_file, mode='a', header=False, index=False)
            else:
                train_loss_data.to_csv(train_loss_csv_file, mode='w', header=True, index=False)

            train_loss = np.average(train_loss)
            
            vali_loss = self.vali(vali_data, vali_loader, criterion, vali_loss_csv_file)

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
                
                avg_time_epoch /= epoch_count
                logger.info(f"Early stopping Epoch: {epoch + 1} Steps: {train_steps} Average_time_epoch {avg_time_epoch:.5f} Min_time_epoch: {min_time_epoch:.5f} Max_time_epoch: {max_time_epoch:.5f}")
                
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion, vali_loss_csv_file):
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                B, T, N = batch_x.shape

                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                loss = self.model(batch_x, None, criterion, None, inp, mask)

                total_loss.append(loss.item())
                
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
        if test:
            print('loading model')
            load_path = os.path.join(self.args.checkpoints, self.args.task_name, self.args.model, 
                                     self.args.data_name,str(self.args.seed), 
                                     self.args.inverse_model,str(self.args.mask_rate), setting)
            self.model.load_state_dict(torch.load(os.path.join(load_path, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        
        folder_path = os.path.join('./test_results', self.args.task_name, self.args.model, 
                                   self.args.data_name, str(self.args.seed), self.args.inverse_model, str(self.timestamp_loss_csv))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(batch_x, None, None, None, inp, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS 
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        
        logger.info(f"Test | mse:{mse:.4f}  mae:{mae:.4f}  rmse:{rmse:.4f}  mape:{mape:.4f}  mspe:{mspe:.4f}")

        return
