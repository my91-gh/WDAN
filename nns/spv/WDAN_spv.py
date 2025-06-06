import os

import torch
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange, repeat
import math
import time
import copy

import torch
import torch.nn as nn
from nns.layers.learnable_dwt import AdaDecomp
from nns.models.iTransformer import Model as iTransformer
from nns.models.PatchTST import Model as PatchTST
from nns.models.Crossformer import Model as Crossformer
from nns.models.FEDformer import Model as FEDformer
from nns.models.WDAN import Model as WDAN
from nns.spv.loss import *
from nns.spv.base_mts_spv import BaseMTSSupervisor

from tqdm import tqdm

Framework_Dict = {
    'WDAN': WDAN,
}

Model_Dict = {
    'iTransformer': iTransformer,
    'PatchTST': PatchTST,
    'Crossformer': Crossformer,
    'FEDformer': FEDformer
}


class WDANSupervisor(BaseMTSSupervisor):
    def __init__(self, **kwargs):
        super(WDANSupervisor, self).__init__(**kwargs)
        self.use_amp = self._train_kwargs.get('precision', 'full') == 'amp'

        self.stats_scaler = GradScaler() if self.use_amp else None
        self.backbone_scaler = GradScaler() if self.use_amp else None
        self.joint_scaler = GradScaler() if self.use_amp else None

        if self.use_amp:
            self.logger.info("Using Automatic Mixed Precision (AMP) training")
        else:
            self.logger.info("Using Full Precision training")

    def _get_model(self):
        framework_name, model_name = self._kwargs['model_id'].split('_')
        self.logger.info(f"Using {framework_name} as framework and {model_name} as model")
        Framework = Framework_Dict[framework_name]
        Model = Model_Dict[model_name]
        self.statistics_pred = Framework(**self._model_kwargs['stats']).to(self.device)
        model = Model(**self._model_kwargs['backbone'])
        return model

    def _reset_stats_optimizer_and_scheduler(self):
        self.stats_optimizer = torch.optim.Adam(self.statistics_pred.parameters(),
                                                lr=self._train_kwargs['base_stats_lr'])
        if self._train_kwargs.get('lr_type', 'MultiStepLR') == 'MultiStepLR':
            self.stats_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.stats_optimizer,
                milestones=self._train_kwargs.get('lr_milestones', [20, 30, 40, 50]),
                gamma=self._train_kwargs.get('lr_decay_ratio', 0.1)
            )
        elif self._train_kwargs.get('lr_type') == 'StepLR':
            self.stats_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.stats_optimizer,
                step_size=int(self._train_kwargs.get('step_size', 1)),
                gamma=self._train_kwargs.get('lr_decay_ratio', 0.5)
            )
        else:
            raise Exception("Unknown lr_type")

    def loss(self, y_pred, y_true):
        if self.loss_type == 'mae':
            criterion = nn.L1Loss()
        elif self.loss_type == 'mse':
            criterion = nn.MSELoss()
        else:
            raise Exception('Unknown loss type')

        pred_loss = criterion(y_pred, y_true)

        if torch.isnan(pred_loss).item():
            self.logger.info('nan occur in loss computation')

        return pred_loss, {
            "pred_loss": pred_loss
        }

    def stats_loss(self, stats_pred, y_true):
        if self.loss_type == 'mae':
            criterion = nn.L1Loss()
        elif self.loss_type == 'mse':
            criterion = nn.MSELoss()
        else:
            raise Exception('Unknown loss type')

        _, yl, yh_s = self.statistics_pred.normalize(y_true, predict=False)
        stats_true = torch.stack([yl, yh_s], dim=1)  # (B, 2, T, N)
        pred_loss = criterion(stats_pred, stats_true)
        if torch.isnan(pred_loss).item():
            self.logger.info('nan occur in loss computation')

        return pred_loss, {
            "pred_loss": pred_loss,
        }

    def model_forward(self, x, y, x_mark, y_mark, pred_only=True):
        x, statistics_pred = self.statistics_pred.normalize(x)
        y_pred = self.model(x, x_mark, y, y_mark)
        y_pred = self.statistics_pred.de_normalize(y_pred, statistics_pred)
        if pred_only:
            return y_pred
        else:
            return y_pred, statistics_pred

    def train(self, trial=None):
        self._reset_stats_optimizer_and_scheduler()

        stats_strategy = self._train_kwargs.get('stats_strategy', 'stats_bb')
        if stats_strategy.startswith('stats'):
            self.logger.info(f"*********Start Step 1 Training*********")
            self._train_stats()

        self.logger.info(f"*********Start Step 2 Training*********")
        if stats_strategy in ['stats_union', 'union']:
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.add_param_group({'params': self.statistics_pred.parameters(), 'lr': lr})
        return self._train(trial)

    def _train_stats(self):
        self.train_stats_iter = 0
        if self._train_kwargs.get('debug', False):
            self.logger.info("Debug Mode")
        best_model = None
        not_improved_count = 0
        best_loss = float('inf')
        train_loss_list = []
        val_loss_list = []

        start_time = time.time()
        for epoch in tqdm(range(1, self._train_kwargs['stats_epochs'] + 1)):
            train_epoch_time = time.time()
            train_epoch_loss = self.train_stats_epoch(epoch)
            epoch_cost_time = time.time() - train_epoch_time
            self.logger.info(f"\tTrain Stats Epoch {epoch} cost time: {epoch_cost_time:.4f}s; "
                             f"Train Stats Speed {epoch_cost_time / self.train_per_epoch:.4f}s/iter")
            if math.isnan(train_epoch_loss):
                self.logger('Found nan loss, training loop break!')
                break
            # learning rate decay
            if self.stats_lr_scheduler is not None:
                self.stats_lr_scheduler.step()

            if self.val_loader is None:
                val_epoch_loss = 0
            else:
                val_epoch_loss = self.val_stats_epoch(epoch)
            self.logger.info(f"\tUpdating LR to: {[group['lr'] for group in self.stats_optimizer.param_groups]}")
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            if self.val_loader is None:
                val_epoch_loss = train_epoch_loss

            # save tensorboard
            if self._save_tb:
                self._writer.add_scalar('loss/train_stats_loss', train_epoch_loss, epoch)
                self._writer.add_scalar('loss/val_stats_loss', val_epoch_loss, epoch)

            # save best state but do not early stop
            if val_epoch_loss < best_loss - 1e-5:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                self.logger.info(f"\tnot_improved_count: {not_improved_count}")
                best_state = False

            # save the best state
            if best_state:
                self.save_stats_checkpoint(epoch)
                best_model = copy.deepcopy(self.statistics_pred.state_dict())
        training_time = time.time() - start_time
        self.logger.info("Total stats training time: {:.4f}min".format(training_time / 60))
        self.statistics_pred.load_state_dict(best_model)
        if self._save_tb:
            self._writer.flush()

    def save_stats_checkpoint(self, epoch):
        if not self._save_and_log:
            self.logger.info("Save_and log is false. No stats models is saved")
            return

        state = {
            'state_dict': self.statistics_pred.state_dict(),
            'optimizer': self.stats_optimizer.state_dict(),
            'config': self._kwargs
        }
        torch.save(state, self._save_model_dir + f'/best_stats_model.pth')
        self.logger.info("Saved current best stats models at {}".format(epoch))

    def load_checkpoint(self, log_dir):
        model_check_point = torch.load(os.path.join(log_dir, 'best_model.pth'))
        stats_check_point = torch.load(os.path.join(log_dir, 'best_stats_model.pth'))
        self.model.load_state_dict(model_check_point['state_dict'])
        self.statistics_pred.load_state_dict(stats_check_point['state_dict'])
        self.logger.info(f'load checkpoint from {log_dir} successfully')

    def train_stats_epoch(self, epoch):
        self.model.eval()
        self.statistics_pred.train()
        total_loss = 0
        total_loss_dict = {}
        for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.train_loader):
            self.train_stats_iter += 1

            x, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)
            self.stats_optimizer.zero_grad()

            if self._train_kwargs.get('debug', False):
                torch.autograd.set_detect_anomaly(True)

            if self.use_amp:  # 自动混合精度
                with autocast():
                    x, statistics_pred = self.statistics_pred.normalize(x)
                    loss, loss_dict = self.stats_loss(statistics_pred, y[:, -self.pred_len:, :])
            else:  # full precision
                x, statistics_pred = self.statistics_pred.normalize(x)
                loss, loss_dict = self.stats_loss(statistics_pred, y[:, -self.pred_len:, :])

            if self._train_kwargs.get('debug', False):
                with torch.autograd.detect_anomaly():
                    if self.use_amp:
                        self.stats_scaler.scale(loss).backward()
                    else:
                        loss.backward()
            else:
                if self.use_amp:
                    self.stats_scaler.scale(loss).backward()
                else:
                    loss.backward()

            if self.use_amp:
                self.stats_scaler.step(self.stats_optimizer)
                self.stats_scaler.update()
            else:
                self.stats_optimizer.step()

            total_loss += loss.item()
            for k in loss_dict.keys():
                if k in total_loss_dict.keys():
                    total_loss_dict[k] += loss_dict[k].item()
                else:
                    total_loss_dict[k] = loss_dict[k].item()

            # log information
            if batch_idx % self._train_kwargs['log_step'] == 0:
                self.logger.info(
                    f'Train Stats Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.3f}')

        train_epoch_loss = total_loss / self.train_per_epoch
        for k in total_loss_dict.keys():
            total_loss_dict[k] = round(total_loss_dict[k] / self.train_per_epoch, 3)
        self.logger.info(f'>>>>>>>>Train Stats Epoch {epoch}: averaged loss {train_epoch_loss:.3f}')
        self.logger.info(f'\tTrain Stats Epoch {epoch}: averaged loss in details {total_loss_dict}')
        return train_epoch_loss

    def val_stats_epoch(self, epoch):
        self.model.eval()
        self.statistics_pred.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.val_loader):
                x, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)

                if self.use_amp:
                    with autocast():
                        x, statistics_pred = self.statistics_pred.normalize(x)
                        loss, _ = self.stats_loss(statistics_pred, y[:, -self.pred_len:, :])
                else:
                    x, statistics_pred = self.statistics_pred.normalize(x)
                    loss, _ = self.stats_loss(statistics_pred, y[:, -self.pred_len:, :])

                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info(f'\tVal Stats Epoch {epoch}: average loss: {val_loss:.6f}')
        return val_loss

    def train_epoch(self, epoch):
        stats_strategy = self._train_kwargs.get('stats_strategy', 'stats_bb')
        if stats_strategy == 'stats_bb':
            self.model.train()
            self.statistics_pred.eval()
            current_scaler = self.backbone_scaler
        elif stats_strategy in ['stats_union', 'union']:
            self.model.train()
            self.statistics_pred.train()
            current_scaler = self.joint_scaler
        elif stats_strategy == 'stats_bb_union':
            twice_epoch = self._train_kwargs.get('twice_epoch', 1)
            if twice_epoch == epoch - 1:
                self.logger.info(f"*********Start Step 3 Training*********")
                lr = self.optimizer.param_groups[0]['lr']
                self.optimizer.add_param_group({'params': self.statistics_pred.parameters(), 'lr': lr})
            self.model.train()
            self.statistics_pred.train()
            current_scaler = self.backbone_scaler if twice_epoch > epoch - 1 else self.joint_scaler
        else:
            raise NotImplementedError

        total_loss = 0
        for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.train_loader):
            self.train_iter += 1

            x, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)
            self.optimizer.zero_grad()

            if self._train_kwargs.get('debug', False):
                torch.autograd.set_detect_anomaly(True)

            if self.use_amp:
                with autocast():
                    y_pred = self.model_forward(x, y, x_mark, y_mark)
                    f_dim = -1 if self._data_kwargs['features'] == 'MS' else 0
                    y_pred, y = y_pred[:, -self.pred_len:, f_dim:], y[:, -self.pred_len:, f_dim:]
                    loss, _ = self.loss(y_pred, y)
            else:
                y_pred = self.model_forward(x, y, x_mark, y_mark)
                f_dim = -1 if self._data_kwargs['features'] == 'MS' else 0
                y_pred, y = y_pred[:, -self.pred_len:, f_dim:], y[:, -self.pred_len:, f_dim:]
                loss, _ = self.loss(y_pred, y)

            if self._train_kwargs.get('debug', False):
                with torch.autograd.detect_anomaly():
                    if self.use_amp:
                        current_scaler.scale(loss).backward()
                    else:
                        loss.backward()
            else:
                if self.use_amp:
                    current_scaler.scale(loss).backward()
                else:
                    loss.backward()

            # add max grad clipping
            if self._train_kwargs.get('clip_grad', False):
                if self.use_amp:
                    current_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._train_kwargs['max_grad_norm'])

            if self.use_amp:
                current_scaler.step(self.optimizer)
                current_scaler.update()
            else:
                self.optimizer.step()

            total_loss += loss.item()

            # log information
            if batch_idx % self._train_kwargs['log_step'] == 0:
                self.logger.info(f'Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.3f}')

        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(f'>>>>>>>>Train Epoch {epoch}: averaged loss {train_epoch_loss:.3f}')
        return train_epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        self.statistics_pred.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.val_loader):
                x, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)

                if self.use_amp:
                    with autocast():
                        y_pred = self.model_forward(x, y, x_mark, y_mark)
                        f_dim = -1 if self._data_kwargs['features'] == 'MS' else 0
                        y_pred, y = y_pred[:, -self.pred_len:, f_dim:], y[:, -self.pred_len:, f_dim:]
                        loss, _ = self.loss(y_pred, y)
                else:
                    y_pred = self.model_forward(x, y, x_mark, y_mark)
                    f_dim = -1 if self._data_kwargs['features'] == 'MS' else 0
                    y_pred, y = y_pred[:, -self.pred_len:, f_dim:], y[:, -self.pred_len:, f_dim:]
                    loss, _ = self.loss(y_pred, y)

                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info(f'\tVal Epoch {epoch}: average loss: {val_loss:.6f}')
        return val_loss
