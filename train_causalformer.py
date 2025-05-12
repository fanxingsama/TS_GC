import torch
from model.TCN_granger.granger_utils import (
    lasso_penalty,
    PGD_update
)


class CausalFormerTrainer:
    def __init__(self, model, epoch, criterion, lr, device, series_num,
                 train_loader, valid_loader, penalty_type, lambda_reg):
        self.model = model
        self.epochs = epoch
        self.criterion = criterion
        self.lr = lr
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.penalty_type = penalty_type
        self.lambda_reg = lambda_reg
        self.series_num = series_num
        self.early_stop = 3
        self.best_MSE_result = float('inf')

    # 一个训练轮次
    def train_epoch(self):
        self.model.train() # 设置模型为训练模式
        first_layer_param = self.model.encoder.layers[0].attention.tcn_processors[0].network_layers[0].conv1.weight
        epoch_loss = 0.0
        epoch_penalty = 0.0
        num_batches = 0
        
        lr_list = [self.lr for _ in range(self.series_num)] # 每个序列的学习率列表
        mse_list = []
        smooth_list = []
        loss_list = []

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # 1. 计算主损失和梯度
            self.model.zero_grad()
            predictions = self.model(batch_x) # 形状: [B, T_out, P, F_out]
            # 确保 target 形状匹配 prediction
            main_loss = self.criterion(predictions, batch_y)
            main_loss.backward() # 计算所有参数的梯度

            epoch_loss += main_loss.item()
            num_batches += 1

            # --- 手动执行近端梯度下降更新 ---
            with torch.no_grad():
                # current_penalty = torch.tensor(0.0, device=self.device)
                current_weights = first_layer_param.data  # 获取当前权重数据
                current_penalty = lasso_penalty(current_weights, self.lambda_reg, self.penalty_type) # 得到Lasso之后的正则化值
                epoch_penalty += current_penalty.item()

                # 更新整个模型的参数
                for name, param in self.model.named_parameters():
                    if param.grad is None: continue # 跳过没有梯度的参数

                    #对第一层使用近端操作符进行近端更新
                    if param is first_layer_param:
                        PGD_update(first_layer_param, self.lambda_reg, self.lr, self.penalty_type) # 近端操作符更新第一层的参数
                        # w_tilde = param.data - self.lr * param.grad  # 梯度下降公式
                        # lambda_gamma = self.lr * self.lambda_reg #  是正则化参数，在近端操作中控制正则化的强度。
                        # w_new = torch.zeros_like(w_tilde)  # 初始化新的权重张量
                        # for j in range(w_tilde.shape[1]): # 遍历输入特征
                        #     w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
                        # if self.penalty_type == 'GL':
                        #     for j in range(w_tilde.shape[1]): # 遍历输入特征
                        #         w_new[:, j, :] = prox_group_lasso(w_tilde[:, j, :], lambda_gamma)
                        # elif self.penalty_type == 'GSGL':
                        #     for j in range(w_tilde.shape[1]):
                        #         w_new[:, j, :] = prox_group_sparse_group_lasso(w_tilde[:, j, :], lambda_gamma, self.alpha_gsgl)
                        # param.copy_(w_new) # 更新参数
                    else:
                        # 对其他参数执行标准梯度下降
                        param.copy_(param.data - self.lr * param.grad)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_epoch_penalty = epoch_penalty / num_batches if num_batches > 0 else float('inf')
            
        return avg_epoch_loss, avg_epoch_penalty

    # 验证轮次
    def valid_epoch(self):
        self.model.eval()
        val_mse = 0.0 # 累计验证集的均方误差
        with torch.no_grad(): # 禁用梯度计算，以提高评估效率。
            for batch_x, batch_y in self.valid_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y) # 验证集损失值
                val_mse += loss.item() * batch_x.size(0) # 乘以批次的大小以得到总的损失值
            avg_val_mse = val_mse / len(self.valid_loader.dataset) if len(self.valid_loader.dataset) > 0 else float('inf')
        return avg_val_mse
    
    # 完整的训练过程
    def train(self):
        not_improved_count = 0  # 未改进计数器，用于记录连续未改进的轮数。
        for epoch in range(1, self.epochs + 1):
            print(f"==第{epoch}轮训练==")
            self.train_epoch()
            result = self.valid_epoch()
            # 如果改进，则更新最佳性能值，重置未改进计数器，并标记为最佳轮次。
            if result <= self.best_MSE_result:  
                self.best_MSE_result = result
                not_improved_count = 0
                print(f"==成功改进，最佳MSE：{result}==")
            else:  # 如果未改进，则增加未改进计数器
                not_improved_count += 1
                print("未改进")
            # 如果未改进计数器超过早停轮数，则停止训练。
            if not_improved_count > self.early_stop:
                print(f"在第{epoch}轮触发早停，模型的最终结果：{result}")
                break
        return self.best_MSE_result
