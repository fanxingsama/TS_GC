import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

# 所有训练器基类
class BaseTrainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        '''
        model：要训练的模型。
        criterion：损失函数。
        metric_ftns：评估指标函数列表。
        optimizer：优化器。
        config：配置对象，包含训练相关的配置信息。
        '''
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # 配置以监视模型性能并最佳保存
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        #  设置可视化编写器实例  
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        # 如果之前保存的有检查点，从检查点恢复训练
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod # 使用 @abstractmethod 装饰器，确保子类必须实现该方法。
    def _train_epoch(self, epoch):
        """
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    # 训练过程中的性能监控
    def train(self):
        not_improved_count = 0 # 未改进计数器，用于记录连续未改进的轮数。
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # 将轮次和训练结果存储到日志字典中
            log = {'epoch': epoch}
            log.update(result)
            for key, value in log.items(): # 遍历日志字典，记录每个键值对到日志。
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # 监控模型的性能
            best = False
            if self.mnt_mode != 'off': # 如果启用了监控模式，检查当前轮次的监控指标是否改进。
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved: # 如果改进，则更新最佳性能值，重置未改进计数器，并标记为最佳轮次。
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else: # 如果未改进，则增加未改进计数器
                    not_improved_count += 1

                # 如果未改进计数器超过早停轮数，则停止训练。
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
             
            # 如果当前轮次是保存周期的倍数，则保存检查点。
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    # 保存检查点
    def _save_checkpoint(self, epoch, save_best=False):
        """
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__ # 获取模型架构名称
        state = { # 构建检查点状态字典，包括模型架构名称、当前轮次、模型状态字典、优化器状态字典、最佳监控指标值和配置对象。
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

    # 从检查点恢复训练
    def _resume_checkpoint(self, resume_path):
        """
        :param resume_path: Checkpoint path to be resumed
        """
        # 读取检查点并写入日志
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        # 加载检查点文件，并更新起始轮次和最佳监控指标值
        checkpoint = torch.load(resume_path, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # 检查检查点中的模型架构配置是否与当前配置一致，不一致就记录警告信息
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict']) # 加载模型状态字典

        # 检查检查点中的优化器类型是否与当前配置一致，不一致就记录警告信息
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer']) # 加载优化器状态字典

        # 记录日志，确认检查点已成功加载
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
