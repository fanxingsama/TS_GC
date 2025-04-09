import numpy as np
import torch
import time
from utils import inf_loop, MetricTracker
from numpy import inf
from logger.logger import get_logger

class Trainer:
    '''
    model：要训练的模型。
    criterion：损失函数。
    metric_ftns：评估指标函数列表。
    optimizer：优化器。
    config：配置对象，包含训练相关的配置信息。
    data_loader：训练数据加载器。
    valid_data_loader：验证数据加载器（可选）。
    lr_scheduler：学习率调度器（可选）。
    lam：正则化项的权重（默认为 0）。
    len_epoch：每个 epoch 的迭代次数（可选，用于迭代式训练）
    '''
    def __init__(self, model, criterion, metric_ftns, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, lam=0, len_epoch=None):
        self.config = config
        self.logger = get_logger('trainer', config['trainer']['verbosity'])

        self.model = model.cuda()
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.early_stop = 10
        self.monitor = "min val_loss"
        
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

        # 从配置文件读取训练器设置
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # 配置以监视模型性能并最佳保存
        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.checkpoint_dir = config.save_dir

        # 跟踪和计算各种指标的平均值
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])


    # 训练过程中的性能监控
    def train(self):
        not_improved_count = 0  # 未改进计数器，用于记录连续未改进的轮数。
        for epoch in range(1, self.epochs + 1):

            print(f"==================第{epoch}轮训练====================")

            result = self._train_epoch(epoch)
            
           # 将轮次和训练结果存储到日志字典中
            log = {'epoch': epoch}
            log.update(result)
            for key, value in log.items():  # 遍历日志字典，记录每个键值对到日志。
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # 监控模型的性能
            best = False

            try:
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False

            if improved:  # 如果改进，则更新最佳性能值，重置未改进计数器，并标记为最佳轮次。
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:  # 如果未改进，则增加未改进计数器
                not_improved_count += 1

            # 如果未改进计数器超过早停轮数，则停止训练。
            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.early_stop))
                break
            
            # 如果当前轮次是保存周期的倍数，则保存检查点。
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    # 模型训练一个epoch
    def _train_epoch(self, epoch):
        self.model.train()  # 设置模型为训练模式
        self.train_metrics.reset()
        epoch_loss = 0.0
        batch_count = 0
        
        start_time = time.time()  # 记录训练周期开始时间
        
        # enumerate:在遍历可迭代对象的时候，同时还获取这个元素的索引到idx里
        for batch_idx, (data, target) in enumerate(self.data_loader):  # 遍历数据加载器
            data, target = data.cuda(), target.cuda()  # 直接使用cuda

            # 模型的的训练过程
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target) + self.lam * self.model.regularization() # 损失函数正则化
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()
            
            # 累加损失值
            epoch_loss += loss.item()
            batch_count += 1
            
            # 更新指标跟踪器
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        
        end_time = time.time()  # 记录训练周期结束时间
        train_time = end_time - start_time  # 计算训练时间
        log = self.train_metrics.result()

        # 如果有验证数据加载器，进行验证
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        # 如果有学习率调度器，更新学习率。
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 打印训练时间
        print(f" 训练时间: {train_time:.2f} s")
        print(f"训练结果: {log}")
        
        
        # 返回日志
        return log
    
    # 训练后进行验证，对模型性能进行评估
    def _valid_epoch(self, epoch):
        self.model.eval() # 设置模型为验证模式
        start_time = time.time()  # 记录验证周期开始时间
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.cuda(), target.cuda()

                # 前向传播，计算损失。
                output = self.model(data)
                loss = self.criterion(output, target)

                # 更新验证指标。
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
        end_time = time.time()  # 记录验证周期结束时间
        val_time = end_time - start_time  # 计算验证时间
        print(f"验证时间: {val_time:.2f} s")
        print(f"验证结果: {self.valid_metrics.result()}")
        
        return self.valid_metrics.result()

    # 保存检查点
    def _save_checkpoint(self, epoch, save_best=False):
        """
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__  # 获取模型架构名称
        state = {  # 构建检查点状态字典，包括模型架构名称、当前轮次、模型状态字典、优化器状态字典、最佳监控指标值和配置对象。
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        # 构建检查点文件名，并保存检查点到文件。
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        # 如果是最佳轮次，则将检查点重命名为 'model_best.pth' 并保存。
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")