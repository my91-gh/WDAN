import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
import time

import torch
import torch.nn as nn
from einops import rearrange

from nns.spv.loss import metrics_np
from nns.spv.base_spv import BaseSupervisor
from nns.spv.run_id import get_model_run_id, get_loss_run_id

from nns.models.iTransformer import Model as iTransformer
from nns.models.PatchTST import Model as PatchTST
from nns.models.Crossformer import Model as Crossformer
from nns.models.FEDformer import Model as FEDformer


Model_Dict = {
    'iTransformer': iTransformer,
    'PatchTST': PatchTST,
    'Crossformer': Crossformer,
    'FEDformer': FEDformer
}


class BaseMTSSupervisor(BaseSupervisor):
    def __init__(self, **kwargs):
        super(BaseMTSSupervisor, self).__init__(**kwargs)

        self.seq_len = self._data_kwargs['seq_len']
        self.pred_len = self._data_kwargs['pred_len']

        self.loss_type = self._loss_kwargs.get('loss_type', 'mae')
        self.mask_value = self._loss_kwargs.get('mask_value', 'none')
        self.logger.info(
            f"loss info: {self.loss_type}, mask value: {self.mask_value}"
        )

    def _gen_run_id(self):
        model_id = self._kwargs.get('model_id')
        seq_len = self._data_kwargs.get('seq_len', 48)
        pred_len = self._data_kwargs.get('pred_len', 96)
        batch_size = self._data_kwargs.get('batch_size')
        run_time = time.strftime('%m%d%H%M%S')
        data_name = self._data_kwargs['data_name']

        loss_settings = get_loss_run_id(self._loss_kwargs, self._train_kwargs)
        model_settings = get_model_run_id(model_id, self._model_kwargs)

        run_id = f'{data_name}_T{seq_len}_H{pred_len}_{model_id}_{run_time}_bs{batch_size}' \
                 f'{model_settings}{loss_settings}'
        return run_id

    def _get_model(self):
        Model = Model_Dict[self._kwargs['model_id']]
        model = Model(**self._model_kwargs)
        return model

    def loss(self, y_pred, y_true):
        """
        input: y_pred, y_true (B, T, N)
        """
        if self.loss_type == 'mae':
            criterion = nn.L1Loss()
        elif self.loss_type == 'mse':
            criterion = nn.MSELoss()
        else:
            raise Exception('Unknown loss type')

        if self.mask_value != 'none':
            mask = torch.gt(y_true.abs(), self.mask_value + 1e-20)
            y_pred = torch.masked_select(y_pred, mask)
            y_true = torch.masked_select(y_true, mask)

        # region series loss calculation
        pred_loss = criterion(y_pred, y_true)
        # endregion

        if torch.isnan(pred_loss).item():
            self.logger.info('nan occur in loss computation')

        return pred_loss, {
            "pred_loss": pred_loss
        }

    def prepare_data(self, x, y, x_mark, y_mark):
        """ split x
        input:
            x: (B, seq_len, N)
            y: (B, label_len + horizon, N)
            x_mark, y_mark: (B, seq_len OR horizon, D)
        output:
            x, xl: (B, seq_len, N)
            y, yl: (B, horizon, N)
            x_mark, y_mark: (B, seq_len OR horizon, D)
        :return:
        """
        x, y = x.float().to(self.device), y.float().to(self.device)
        if x_mark is not None:
            x_mark, y_mark = x_mark.float().to(self.device), y_mark.float().to(self.device)
        return x, y, x_mark, y_mark

    def model_forward(self, x, y, x_mark, y_mark):
        y_pred = self.model(x, x_mark, y, y_mark)
        return y_pred

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.train_loader):
            self.train_iter += 1

            x, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)

            self.optimizer.zero_grad()

            if self._train_kwargs.get('debug', False):
                torch.autograd.set_detect_anomaly(True)

            y_pred = self.model_forward(x, y, x_mark, y_mark)
            f_dim = -1 if self._data_kwargs['features'] == 'MS' else 0
            y_pred, y = y_pred[:, -self.pred_len:, f_dim:], y[:, -self.pred_len:, f_dim:]
            loss, _ = self.loss(y_pred, y)

            if self._train_kwargs.get('debug', False):
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()

            # add max grad clipping
            if self._train_kwargs.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._train_kwargs['max_grad_norm'])

            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self._train_kwargs['log_step'] == 0:
                self.logger.info(f'Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.3f}')\

        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(f'>>>>>>>>Train Epoch {epoch}: averaged loss {train_epoch_loss:.3f}')
        return train_epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.val_loader):
                x, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)
                y_pred = self.model_forward(x, y, x_mark, y_mark)
                f_dim = -1 if self._data_kwargs['features'] == 'MS' else 0
                y_pred, y = y_pred[:, -self.pred_len:, f_dim:], y[:, -self.pred_len:, f_dim:]
                loss, _ = self.loss(y_pred, y)

                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info(f'\tVal Epoch {epoch}: average loss: {val_loss:.6f}')
        return val_loss

    def test(self, test_loader=None, message=None):
        if test_loader is None:
            test_loader = self.test_loader
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch_idx, (x, y, x_mark, y_mark) in enumerate(test_loader):
                x, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)
                y_pred = self.model_forward(x, y, x_mark, y_mark)

                y_pred, y = y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()

                preds.append(y_pred)
                labels.append(y)

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        if self.data['test_data'].scale and self._test_kwargs.get('inverse', False):
            print("Inverse Transform Test Predictions...")
            preds = rearrange(preds, 'b t n -> (b t) n')
            labels = rearrange(labels, 'b t n -> (b t) n')
            preds = rearrange(self.data['test_data'].inverse_transform(preds), '(b t) n -> b t n', t=self.pred_len)
            labels = rearrange(self.data['test_data'].inverse_transform(labels), '(b t) n -> b t n', t=self.pred_len)

        f_dim = -1 if self._data_kwargs['features'] == 'MS' else 0
        preds, labels = preds[:, -self.pred_len:, f_dim:], labels[:, -self.pred_len:, f_dim:]
        print(f"y_preds {preds.shape}; labels {labels.shape}")

        if self._test_kwargs.get('key_hors', None) is None:
            key_hor_list = [0] + np.arange(47, self.pred_len, 48).tolist()
        else:
            key_hor_list = [_ - 1 for _ in self._kwargs['test']['key_hors']]

        # region loss dict
        mask_value = self._test_kwargs.get('mask_value', 'none')
        mae_overall, mse_overall, rmse_overall, smape_overall, r2_overall = metrics_np(preds, labels, mask_value)
        loss_dict = {
            'mae_overall': mae_overall,
            'mse_overall': mse_overall,
            'rmse_overall': rmse_overall,
            'smape_overall': smape_overall,
            'r2_overall': r2_overall
        }
        self.logger.info(f"Horizon Average: MSE {mse_overall:.3f}, MAE {mae_overall:.3f}, R2 {r2_overall:.3f}")
        for hor in key_hor_list:
            mae_hor, mse_hor, rmse_hor, smape_hor, r2_hor = metrics_np(preds[:, hor, :], labels[:, hor, :], mask_value)
            loss_dict[f'hor_{hor + 1}'] = {
                'mae': mae_hor, 'mse': mse_hor, 'rmse': rmse_hor, 'smape': smape_hor, 'r2': r2_hor
            }
            self.logger.info(f"Horizon {hor + 1}: MSE {mse_hor:.3f}, MAE {mae_hor:.3f}, R2 {r2_hor:.3f}")
        # endregion

        # region save test results
        # quick save
        settings = self._run_id if message is None else self._run_id + '_' + message
        model_id = self._kwargs.get('model_id')
        data_name = self._data_kwargs['data_name']
        local_flag = False if 'root' in self.log_dir else True
        file_name = f'result_{model_id}_{data_name}.txt' if local_flag else f'result_{model_id}_{data_name}_server.txt'
        if not os.path.exists('quick_rets'):
            os.makedirs('quick_rets')
        with open('quick_rets/' + file_name, 'a') as f:
            f.write(settings + '\n')
            f.write(f"Horizon Average: MSE {mse_overall:.4f}, MAE {mae_overall:.4f}, R2 {r2_overall:.4f}\n\n")

        # save in log_dir
        if self._save_and_log:
            save_dict = {
                'settings': settings,
                'rets': loss_dict
            }
            with open(os.path.join(self.log_dir, 'test_record.txt'), 'w') as f:
                json.dump(save_dict, f, indent=2)
        # endregion

        return {
            'test_data': (labels, preds),
            'loss': loss_dict
        }











