import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import utils.loss_metric as module_metric
import model.model as module_arch
from utils.args_config_analyse import args_config_analyse
from trainer import Trainer
from utils import prepare_device

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train') # 日志记录器

    data_loader = config.init_obj('data_loader', module_data) # 初始化数据加载器
    valid_data_loader = data_loader.split_validation() # 分离出验证集
    config['data_loader']['args']['series_num'] = data_loader.series_num # 设置数据加载器参数
    config['data_loader']['args']['time_step'] = data_loader.time_step 
    config['data_loader']['args']['output_window'] = data_loader.output_window
    
    model = config.init_obj('arch', module_arch, config) # 构建模型架构
    logger.info(model) # 打印模型架构

    device = prepare_device() # 准备训练模型的设备
    model = model.to(device) # 将模型加载到设备上
    criterion = getattr(module_metric, config['loss']) # 获取所需要使用的损失函数
    metrics = [getattr(module_metric, met) for met in config['metrics']] # 获取所需要使用的评估指标

    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) # 获取需要训练的参数
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params) # 构建优化器
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer) # 构建学习率调整器
    lam = config['trainer']['lam'] # 获取训练相关参数

    # 开始训练模型
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      lam = lam)

    trainer.train()
    print('===============模型训练结束==============')


# 单独运行这个文件
if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Causality')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = args_config_analyse.from_args(args=args) # 根据命令行参数和自定义选项，解析配置文件并生成配置对象
    main(config)
