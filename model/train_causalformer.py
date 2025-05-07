import argparse
import json
import os
import torch
import numpy as np
import pandas as pd

# 根据您的项目结构以及运行脚本的方式调整导入。
# 如果从 causalFormer/ 目录运行 `python model/train_causalFormer.py`，
# 并且 causalFormer/ 在 PYTHONPATH 中，或者您通过 sys.path 添加它。
# 这些导入假设 `causalFormer` 是顶级包。
from causalFormer.logger.logger import get_logger # 确保此路径正确
from causalFormer.data_loaders import data_provider
from causalFormer.model.Granger_causalFormer import GrangerCausalFormer
import causalFormer.model.loss_metric as module_loss
import causalFormer.model.loss_metric as module_metric

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 默认随机种子
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
# torch.backends.cudnn.deterministic = True # 可能会降低速度，如果需要则启用
# torch.backends.cudnn.benchmark = False   # 如果输入大小不变则启用

class CausalFormerTrainer:
    """
    负责CausalFormer模型训练、验证、测试和因果矩阵生成的类。
    """
    def __init__(self, model, criterion, metrics_fns, optimizer, config, device,
                 train_loader, valid_loader, test_loader, n_series,
                 lr_scheduler=None, logger=None):
        self.model = model
        self.criterion = criterion
        self.metrics_fns = metrics_fns
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.logger = logger or get_logger(config.get('name', 'CausalFormer_Trainer_Default')) # 提供一个默认记录器

        # 从配置中提取训练参数
        trainer_conf = self.config['trainer']
        self.epochs = trainer_conf['epochs']
        self.save_dir = trainer_conf['save_dir']
        self.monitor_mode = trainer_conf.get('monitor_mode', 'min')
        self.monitor_metric_name = trainer_conf.get('monitor_metric', 'val_loss') # 假设监控验证损失
        self.save_period = trainer_conf.get('save_period')
        self.early_stop = trainer_conf.get('early_stop', float('inf'))
        self.grad_norm_clip = trainer_conf.get('grad_norm_clip', 1.0)

        self.best_val_metric = float('inf' if self.monitor_mode == 'min' else '-inf')
        self.epochs_no_improve = 0
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"模型将保存在: {self.save_dir}")

        # 从数据加载器配置中提取序列长度信息
        dl_conf = self.config['data_loader']['args']
        self.seq_len = dl_conf['size'][0]
        self.label_len = dl_conf['size'][1] # 解码器的起始标记长度
        self.pred_len = dl_conf['size'][2] # 预测范围

        self.n_series = n_series # 从外部传入，因为数据加载器创建后才知道
        self.model_args = self.config['arch']['args'] # 保存模型参数以备后用

    def _train_epoch(self, epoch):
        """
        执行一个训练轮次。
        :param epoch: 当前轮次编号
        :return: 平均训练损失和训练指标字典
        """
        self.model.train()
        total_train_loss = 0
        train_metrics_agg = {m.__name__: 0.0 for m in self.metrics_fns}

        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            dec_inp_label_part = batch_y[:, :self.label_len, :]
            dec_inp_pred_part = torch.zeros_like(batch_y[:, self.label_len:(self.label_len + self.pred_len), :]).float()
            dec_inp = torch.cat([dec_inp_label_part, dec_inp_pred_part], dim=1).float().to(self.device)

            self.optimizer.zero_grad()
            
            if self.model_args.get('output_attention', False):
                outputs, attention = self.model(batch_x, dec_inp) 
            else:
                outputs = self.model(batch_x, dec_inp)

            targets = batch_y[:, self.label_len:(self.label_len + self.pred_len), :].to(self.device)
            
            if outputs.shape != targets.shape:
                self.logger.warning(f"训练中形状不匹配！输出: {outputs.shape}, 目标: {targets.shape}。")
                # outputs = outputs[:, -self.pred_len:, :] # 调整逻辑可能需要根据模型具体行为

            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            self.optimizer.step()

            total_train_loss += loss.item()
            for met_fn in self.metrics_fns:
                train_metrics_agg[met_fn.__name__] += met_fn(outputs.detach(), targets.detach()).item() * batch_x.size(0)
        
        avg_train_loss = total_train_loss / len(self.train_loader)
        for name in train_metrics_agg:
            train_metrics_agg[name] /= len(self.train_loader.dataset)
            
        return avg_train_loss, train_metrics_agg

    def _valid_epoch(self, epoch):
        """
        执行一个验证轮次。
        :param epoch: 当前轮次编号
        :return: 平均验证损失和验证指标字典
        """
        self.model.eval()
        total_val_loss = 0
        val_metrics_agg = {m.__name__: 0.0 for m in self.metrics_fns}
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.valid_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                dec_inp_label_part = batch_y[:, :self.label_len, :]
                dec_inp_pred_part = torch.zeros_like(batch_y[:, self.label_len:(self.label_len + self.pred_len), :]).float()
                dec_inp = torch.cat([dec_inp_label_part, dec_inp_pred_part], dim=1).float().to(self.device)

                if self.model_args.get('output_attention', False):
                    outputs, attention = self.model(batch_x, dec_inp)
                else:
                    outputs = self.model(batch_x, dec_inp)
                
                targets = batch_y[:, self.label_len:(self.label_len + self.pred_len), :].to(self.device)

                if outputs.shape != targets.shape:
                    self.logger.warning(f"验证中形状不匹配！输出: {outputs.shape}, 目标: {targets.shape}。")
                    # outputs = outputs[:, -self.pred_len:, :]

                loss = self.criterion(outputs, targets)
                total_val_loss += loss.item()
                for met_fn in self.metrics_fns:
                    val_metrics_agg[met_fn.__name__] += met_fn(outputs, targets).item() * batch_x.size(0)
        
        avg_val_loss = total_val_loss / len(self.valid_loader)
        for name in val_metrics_agg:
            val_metrics_agg[name] /= len(self.valid_loader.dataset)
            
        return avg_val_loss, val_metrics_agg

    def train(self):
        """
        执行完整的训练过程。
        """
        for epoch in range(1, self.epochs + 1):
            avg_train_loss, train_metrics = self._train_epoch(epoch)
            
            log_msg = f"轮次 {epoch}/{self.epochs} - 训练损失: {avg_train_loss:.4f}"
            for name, val in train_metrics.items():
                log_msg += f" - 训练 {name}: {val:.4f}"
            self.logger.info(log_msg)

            avg_val_loss, val_metrics = self._valid_epoch(epoch)
            log_msg_val = f"轮次 {epoch}/{self.epochs} - 验证损失: {avg_val_loss:.4f}"
            for name, val in val_metrics.items():
                log_msg_val += f" - 验证 {name}: {val:.4f}"
            self.logger.info(log_msg_val)

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                    # 根据配置决定监控哪个指标
                    metric_to_monitor = avg_val_loss # 默认
                    if self.monitor_metric_name != 'val_loss' and self.monitor_metric_name in val_metrics:
                        metric_to_monitor = val_metrics[self.monitor_metric_name]
                    self.lr_scheduler.step(metric_to_monitor)
                else:
                    self.lr_scheduler.step()

            # 确定当前轮次的监控指标值
            current_monitored_val = avg_val_loss # 默认
            if self.monitor_metric_name != 'val_loss' and self.monitor_metric_name in val_metrics:
                current_monitored_val = val_metrics[self.monitor_metric_name]
            
            improved = (self.monitor_mode == 'min' and current_monitored_val < self.best_val_metric) or \
                       (self.monitor_mode == 'max' and current_monitored_val > self.best_val_metric)

            if improved:
                self.best_val_metric = current_monitored_val
                self.epochs_no_improve = 0
                model_save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), model_save_path)
                self.logger.info(f"保存最佳模型到 {model_save_path} (监控指标 {self.monitor_metric_name}: {self.best_val_metric:.4f})")
            else:
                self.epochs_no_improve += 1

            if self.save_period and epoch % self.save_period == 0:
                model_save_path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pth")
                torch.save(self.model.state_dict(), model_save_path)
                self.logger.info(f"保存检查点到 {model_save_path}")

            if self.epochs_no_improve >= self.early_stop:
                self.logger.info(f"在 {self.epochs_no_improve} 个轮次没有改进后触发早停。")
                break
        
        self._test()
        self._generate_causality_matrix()
        self.logger.info("训练和评估完成。")

    def _test(self):
        """
        在测试集上评估最佳模型。
        """
        self.logger.info("加载最佳模型以在测试集上进行最终评估...")
        best_model_path = os.path.join(self.save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            self.logger.warning("未找到 best_model.pth。使用最后一个模型状态进行测试。")

        self.model.eval()
        total_test_loss = 0
        test_metrics_agg = {m.__name__: 0.0 for m in self.metrics_fns}
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                dec_inp_label_part = batch_y[:, :self.label_len, :]
                dec_inp_pred_part = torch.zeros_like(batch_y[:, self.label_len:(self.label_len + self.pred_len), :]).float()
                dec_inp = torch.cat([dec_inp_label_part, dec_inp_pred_part], dim=1).float().to(self.device)
                
                if self.model_args.get('output_attention', False):
                    outputs, attention = self.model(batch_x, dec_inp)
                else:
                    outputs = self.model(batch_x, dec_inp)

                targets = batch_y[:, self.label_len:(self.label_len + self.pred_len), :].to(self.device)
                
                if outputs.shape != targets.shape:
                    self.logger.warning(f"测试中形状不匹配！输出: {outputs.shape}, 目标: {targets.shape}。")
                    # outputs = outputs[:, -self.pred_len:, :]

                loss = self.criterion(outputs, targets)
                total_test_loss += loss.item()
                for met_fn in self.metrics_fns:
                   test_metrics_agg[met_fn.__name__] += met_fn(outputs, targets).item() * batch_x.size(0)

        avg_test_loss = total_test_loss / len(self.test_loader)
        log_msg_test = f"测试结果 - 损失: {avg_test_loss:.4f}"
        for name, val in test_metrics_agg.items():
            test_metrics_agg[name] /= len(self.test_loader.dataset)
            log_msg_test += f" - {name}: {test_metrics_agg[name]:.4f}"
        self.logger.info(log_msg_test)

    def _generate_causality_matrix(self):
        """
        计算并保存格兰杰因果关系矩阵。
        """
        self.logger.info("正在计算格兰杰因果关系矩阵...")
        significance_level = self.config.get('causality_params', {}).get('significance_level', 0.05)
        
        causal_matrix = self.model.get_granger_causality_matrix(significance_level=significance_level)
        
        if causal_matrix is not None:
            if isinstance(causal_matrix, torch.Tensor):
                causal_matrix_np = causal_matrix.detach().cpu().numpy()
            else:
                causal_matrix_np = causal_matrix
                
            causal_matrix_path_npy = os.path.join(self.save_dir, "granger_causality_matrix.npy")
            np.save(causal_matrix_path_npy, causal_matrix_np)
            self.logger.info(f"格兰杰因果关系矩阵 (NumPy 数组) 已保存到 {causal_matrix_path_npy}")

            if len(causal_matrix_np.shape) == 2 and causal_matrix_np.shape[0] == self.n_series and causal_matrix_np.shape[1] == self.n_series:
                try:
                    series_names = getattr(self.train_loader.dataset, 'feature_names', [f'series_{i}' for i in range(self.n_series)])
                    if len(series_names) != self.n_series:
                        series_names = [f'series_{i}' for i in range(self.n_series)]

                    df_causal_matrix = pd.DataFrame(causal_matrix_np, index=series_names, columns=series_names)
                    causal_matrix_path_csv = os.path.join(self.save_dir, "granger_causality_matrix.csv")
                    df_causal_matrix.to_csv(causal_matrix_path_csv)
                    self.logger.info(f"格兰杰因果关系矩阵也已另存为 CSV 到 {causal_matrix_path_csv}")
                except Exception as e:
                    self.logger.error(f"无法将格兰杰因果关系矩阵另存为 CSV: {e}")
            elif len(causal_matrix_np.shape) != 2:
                 self.logger.warning(f"格兰杰因果关系矩阵不是二维的 (形状: {causal_matrix_np.shape})，无法直接另存为 N,N CSV。")
            else:
                self.logger.warning(f"格兰杰因果关系矩阵是二维的，但形状 {causal_matrix_np.shape} 与 (N={self.n_series}, N={self.n_series}) 不匹配。不另存为 CSV。")
        else:
            self.logger.warning("格兰杰因果关系矩阵无法计算或模型返回为 None。")


def main(config):
    # 设置记录器
    run_name = config.get('name', 'CausalFormer_Run')
    logger = get_logger(run_name)
    logger.info(f"开始运行: {run_name}")
    logger.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}") # ensure_ascii=False 用于正确显示中文

    # 设置设备
    use_cuda = config.get('n_gpu', 0) > 0 and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        if config.get('n_gpu', 0) > torch.cuda.device_count():
            logger.warning(f"警告: n_gpu ({config.get('n_gpu',0)}) 大于可用 GPU 数量 ({torch.cuda.device_count()})")
    else:
        device = torch.device("cpu")
        logger.info("使用 CPU")

    # 数据加载
    dl_conf = config['data_loader']['args']
    logger.info(f"数据加载器配置: {dl_conf}")
    
    if 'pred_len' not in config['arch']['args']:
        config['arch']['args']['pred_len'] = dl_conf['size'][2]
    elif config['arch']['args']['pred_len'] != dl_conf['size'][2]:
        logger.warning(f"模型 pred_len ({config['arch']['args']['pred_len']}) 和数据加载器 pred_len ({dl_conf['size'][2]}) 不匹配。使用模型的设置。")

    train_loader = data_provider(dl_conf, 'train')
    valid_loader = data_provider(dl_conf, 'val')
    test_loader = data_provider(dl_conf, 'test')
    logger.info("数据加载器已创建。")

    n_series = train_loader.dataset.data_x.shape[-1]
    logger.info(f"序列数量 (特征数): {n_series}")

    # 模型
    model_args = config['arch']['args']
    model_args['n_series'] = n_series
    model_args['lag'] = dl_conf['size'][0]
    
    logger.info(f"模型参数: {model_args}")
    model = GrangerCausalFormer(**model_args).to(device)

    # 损失函数和评估指标
    criterion_config = config.get('loss', {'type': 'MSELoss', 'args': {}})
    if hasattr(torch.nn, criterion_config['type']):
        criterion = getattr(torch.nn, criterion_config['type'])(**criterion_config.get('args', {}))
    else:
        criterion = getattr(module_loss, criterion_config['type'])(**criterion_config.get('args', {}))
    logger.info(f"损失函数: {criterion_config['type']}")

    metrics_conf = config.get('metrics', [])
    metrics_fns = []
    for met_name in metrics_conf:
        if hasattr(module_metric, met_name):
            metrics_fns.append(getattr(module_metric, met_name))
        else:
            logger.warning(f"指标 {met_name} 在自定义指标模块中未找到。")
    logger.info(f"评估指标: {[fn.__name__ for fn in metrics_fns]}")

    # 优化器和学习率调度器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_conf = config['optimizer']
    optimizer = getattr(optim, optimizer_conf['type'])(trainable_params, **optimizer_conf['args'])
    logger.info(f"优化器: {optimizer_conf['type']}，参数: {optimizer_conf['args']}")

    scheduler = None
    if 'lr_scheduler' in config and config['lr_scheduler']['type'] is not None:
        scheduler_conf = config['lr_scheduler']
        scheduler = getattr(lr_scheduler, scheduler_conf['type'])(optimizer, **scheduler_conf['args'])
        logger.info(f"学习率调度器: {scheduler_conf['type']}，参数: {scheduler_conf['args']}")

    # 初始化并运行训练器
    trainer = CausalFormerTrainer(
        model=model,
        criterion=criterion,
        metrics_fns=metrics_fns,
        optimizer=optimizer,
        config=config,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        n_series=n_series,
        lr_scheduler=scheduler,
        logger=logger
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CausalFormer 训练脚本')
    parser.add_argument('-c', '--config', default=None, type=str, required=True,
                        help='JSON 配置文件的路径 (必需)')
    parser.add_argument('--device', default=None, type=str,
                        help='指定 GPU 设备 ID，例如 "0" 或 "0,1"。如果设置，则覆盖配置中的 n_gpu。')

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f: # 指定 utf-8 编码
        config_data = json.load(f)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        config_data['n_gpu'] = len(args.device.split(','))

    main(config_data)
