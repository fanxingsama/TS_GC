import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
torch.autograd.set_detect_anomaly(True)
from sklearn.cluster import KMeans

class Trainer(BaseTrainer):
    '''
    model：要训练的模型。
    criterion：损失函数。
    metric_ftns：评估指标函数列表。
    optimizer：优化器。
    config：配置对象，包含训练参数。
    device：设备（CPU 或 GPU）。
    data_loader：训练数据加载器。
    valid_data_loader：验证数据加载器（可选）。
    lr_scheduler：学习率调度器（可选）。
    lam：正则化项的权重（默认为 0）。
    len_epoch：每个 epoch 的迭代次数（可选，用于迭代式训练）。
    '''
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, lam=0, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        # 根据 len_epoch 是否为 None，决定是基于 epoch 还是基于迭代进行训练。
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.lam = lam
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    # 模型训练
    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        # enumerate:在遍历可迭代对象的时候，同时还获取这个元素的索引到idx里
        for batch_idx, (data, target) in enumerate(self.data_loader): # 遍历数据加载器
            data, target = data.to(self.device), target.to(self.device) # 将数据移动到指定设备上

            # 模型的的训练过程
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target) + self.lam * self.model.regularization()
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()
            
            # 
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # 每隔 log_step 批次，记录日志。
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            
            # 如果达到 len_epoch，停止训练。
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        # 如果有验证数据加载器，进行验证
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        # 如果有学习率调度器，更新学习率。
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 返回日志
        return log
    
    # 训练后进行验证，对模型性能进行评估
    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播，计算损失。
                output = self.model(data)
                loss = self.criterion(output, target)

                # 更新验证指标。
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # 将模型参数的直方图添加到 TensorBoard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    # 计算并返回当前进度的字符串表示。 
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
