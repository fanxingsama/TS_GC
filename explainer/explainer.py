import torch

# Regression Relevance Propogation
class RRP:
    def __init__(self, model):
        self.model = model
        self.model.eval() # 模型设置为评估模式

    # 为整个输入数据生成回归相关性分数
    def generate_RRP(self, batch_size, input, interpreted_series):
        inputs = torch.split(input, batch_size) # split inputs into batches
        relAs, relKs = [], []
        for data in inputs:
            relA, relK = self._generate_RRP(data, interpreted_series) # generate RRP for each batch
            relAs.append(relA)
            relKs.append(relK)
        relA = torch.stack(relAs).mean(0)
        relK = torch.stack(relKs).mean(0)
        return relA, relK # 返回注意矩阵和卷积核的因果分数
        
    # 为单个批次的输入数据生成因果关系分数。
    def _generate_RRP(self, input, interpreted_series):
        """
        input (torch.Tensor):输入数据张量[total_batch， input_window, series_num, feature_dim]
        interpreted_series (int)：被解释时间序列的索引。
        """
        # Forward pass through the model
        output = self.model(input)
        # 创建一个与输出形状相同的 one-hot 张量，仅在 interpreted_series 对应的位置设置为 1。
        one_hot = torch.zeros_like(output, dtype=torch.float).to(output.device)
        one_hot[:,:,interpreted_series,:] = 1
        # 克隆 one-hot 张量，并设置其需要计算梯度。
        one_hot_vector = one_hot.clone()
        one_hot.requires_grad_(True)
        # 计算 one-hot 张量与模型输出的点积。
        one_hot = torch.sum(one_hot * output)
        # 清空模型的梯度，并进行反向传播。
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # 调用模型的 relprop 方法，将 one-hot 张量的梯度传播回输入层，计算每个输入特征对输出的贡献
        self.model.relprop(one_hot_vector)

        # 收集因果关系分数
        relAs=[] # 注意力矩阵因果关系分数
        relKs=[] # 卷积核因果关系分数
        for layer in self.model.encoder.layers: # 遍历模型的编码器层，收集每个层的因果关系分数。
            # 梯度调制
            relA = layer.attention.attention.get_rel() * torch.abs(layer.attention.attention.get_grad())
            relK = layer.attention.Wv.get_rel() * torch.abs(layer.attention.Wv.get_grad())

            # w/o interpretation
            # relA = layer.attention.attention.get_wgt()
            # relK = layer.attention.Wv.get_wgt()

            relA = relA.clamp(min=0)        # 只考虑正向因果分数
            relK = relK.clamp(min=0)        
            relAs.append(relA.mean((0,1)))  # mean for sample and head
            relKs.append(relK.mean(0))      # mean for head
        # 将所有层的分数堆叠起来，并计算它们的乘积，得到最终的因果关系分数。
        relA = torch.stack(relAs).prod(0)   
        relK = torch.stack(relKs).prod(0)  
        return relA, relK