import argparse
from pathlib import Path
import torch
import numpy as np
import data_loader.data_loaders as module_data
import utils.loss_metric as module_metric
import model.model as module_arch
from datetime import datetime
from logger.logger import get_logger, setup_logging
import time
from utils import inf_loop, init_obj_by_config, from_args, write_json
from numpy import inf

from utils.util import read_json

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

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
                 data_loader, logger, run_id, valid_data_loader=None, lr_scheduler=None, lam=0, len_epoch=None):
        self.config = config
        self.logger = logger
        self.model = model.cuda()
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.run_id = run_id
        self.data_loader = data_loader
        self.early_stop = 10
        
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

        # 配置以监视模型性能并最佳保存
        self.mnt_best = inf

        # 跟踪和计算各种指标的平均值
        self.train_metrics = {'loss': 0.0}  # 初始化训练指标
        self.valid_metrics = {'loss': 0.0}  # 初始化验证指标
        for m in metric_ftns:
            self.train_metrics[m.__name__] = 0.0
            self.valid_metrics[m.__name__] = 0.0
        
         # 添加计数器和总样本数用于计算平均值
        self.train_count = 0
        self.valid_count = 0


    # 训练过程中的性能监控
    def train(self):
        not_improved_count = 0  # 未改进计数器，用于记录连续未改进的轮数。
        for epoch in range(1, self.epochs + 1):
            print(f"==================第{epoch}轮训练====================")
            result = self._train_epoch(epoch)
            # 监控模型的性能
            best = False
            improved = self.train_metrics['val_loss'] <= self.mnt_best
            if improved:  # 如果改进，则更新最佳性能值，重置未改进计数器，并标记为最佳轮次。
                self.mnt_best = self.train_metrics['val_loss']
                not_improved_count = 0
                best = True
            else:  # 如果未改进，则增加未改进计数器
                not_improved_count += 1
            # 如果未改进计数器超过早停轮数，则停止训练。
            if not_improved_count > self.early_stop:
                self.logger.info(f"一共训练了{epoch}轮，模型的最终结果：{result}")
                break
            # 保存检查点。
            if epoch % 1 == 0:
                self._save_checkpoint(epoch, save_best=best)
    # 模型训练一个epoch
    def _train_epoch(self, epoch):
        self.model.train()  # 设置模型为训练模式
        # 重置指标和计数器
        self.train_metrics = {k: 0.0 for k in self.train_metrics}
        self.train_count = 0
        
        # self.train_metrics.reset()
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
            batch_size = data.size(0)
            self.train_count += batch_size
            self.train_metrics['loss'] += loss.item() * batch_size
            for met in self.metric_ftns:
                self.train_metrics[met.__name__] += met(output, target) * batch_size
        
        
        # 计算平均指标
        for k in self.train_metrics:
            self.train_metrics[k] /= self.train_count
            
        end_time = time.time()  # 记录训练周期结束时间
        train_time = end_time - start_time  # 计算训练时间

        # 验证（如果存在验证集）
        if self.do_validation:
            val_log = self._valid_epoch()
            # 将验证指标合并到训练指标，并添加 "val_" 前缀
            for k, v in val_log.items():
                self.train_metrics[f'val_{k}'] = v

        # 如果有学习率调度器，更新学习率。
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 打印训练时间
        print(f" 训练时间: {train_time:.2f} s")
        print(f"训练结果: {self.train_metrics}")
        
        
        # 返回日志
        return self.train_metrics  # 直接返回字典
    
    # 训练后进行验证，对模型性能进行评估
    def _valid_epoch(self):
        self.model.eval() # 设置模型为验证模式
         # 重置验证指标和计数器
        self.valid_metrics = {k: 0.0 for k in self.valid_metrics}
        self.valid_count = 0
        
        start_time = time.time()  # 记录验证周期开始时间

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.cuda(), target.cuda()

                # 前向传播，计算损失。
                output = self.model(data)
                loss = self.criterion(output, target)

                # 更新验证指标（累加）
                batch_size = data.size(0)
                self.valid_count += batch_size
                self.valid_metrics['loss'] += loss.item() * batch_size
                for met in self.metric_ftns:
                    self.valid_metrics[met.__name__] += met(output, target) * batch_size
        # 计算平均指标
        for k in self.valid_metrics:
            self.valid_metrics[k] /= self.valid_count
            
        end_time = time.time()  # 记录验证周期结束时间
        val_time = end_time - start_time  # 计算验证时间
        print(f"验证时间: {val_time:.2f} s")
        print(f"验证结果: {self.valid_metrics}")
        
        return self.valid_metrics  # 返回验证指标字典

    # 保存检查点
    def _save_checkpoint(self, epoch, save_best=False):
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
        filename = Path('saved') / self.run_id / 'model' / 'checkpoint-epoch{}.pth'.format(epoch)
        torch.save(state, filename)

        # 如果是最佳轮次，则将检查点重命名为 'model_best.pth' 并保存。
        if save_best:
            best_path = Path('saved') / self.run_id  / 'model' / 'model_best.pth'
            torch.save(state, best_path)

def main(config, run_id):
    log_save_path = Path('saved') / run_id / 'train_result_log'
    filename = Path('saved') / run_id / 'model'
    filename.mkdir(parents=True, exist_ok=True)
    setup_logging(log_save_path)
    train_logger = get_logger() # 日志记录器
    write_json(config, Path('saved') / run_id / 'model_config.json') # 模型超参数保存
    
    data_loader = init_obj_by_config(config, 'data_loader', module_data) # 初始化数据加载器
    valid_data_loader = data_loader.split_validation() # 分离出验证集
    
    config['data_loader']['args']['series_num'] = data_loader.series_num # 设置数据加载器参数
    config['data_loader']['args']['time_step'] = data_loader.time_step 
    config['data_loader']['args']['output_window'] = data_loader.output_window
    
    model = init_obj_by_config(config, 'model', module_arch, config) # 构建模型架构,最后加个config是因为model在初始化的时候需要这个参数
    train_logger.info(model) # 模型架构保存
    train_logger.info("==============模型训练开始==============") # 打印模型架构
    model = model.cuda() # 将模型加载到设备上
    criterion = getattr(module_metric, config['loss']) # 获取所需要使用的损失函数
    metrics = [getattr(module_metric, met) for met in config['metrics']] # 获取所需要使用的评估指标

    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) # 获取需要训练的参数
    optimizer = init_obj_by_config(config, 'optimizer', torch.optim, trainable_params) # 构建优化器
    lr_scheduler = init_obj_by_config(config, 'lr_scheduler', torch.optim.lr_scheduler, optimizer) # 构建学习率调整器
    lam = config['trainer']['lam'] # 获取训练相关参数

    # 开始训练模型
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      logger= train_logger,
                      run_id=run_id,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      lam = lam)

    trainer.train()
    print('===============模型训练结束==============')


# 单独运行这个文件
if __name__ == '__main__':
    run_id = datetime.now().strftime(r'%m%d_%H%M%S') # 获得当前时间
    config_path = 'config/config_demo.json'
    config = read_json(Path(config_path))
    main(config, run_id)
    torch.cuda.empty_cache() # 清理显存