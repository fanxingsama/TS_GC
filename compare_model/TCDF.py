import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import ADDSTCN
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys

# 数据准备，从csv里拿出数据，组织一下，然后转为torch能够处理的格式
def preparedata(file, target):
    df_data = pd.read_csv(file)
    df_y = df_data.copy(deep=True)[[target]] # 目标时间序列 y，使用[[target]]直接让df_y即使只有一列，但也能变成一个DataFrame，而不是series
    df_x = df_data.copy(deep=True) # 输入序列，x是有多列的，所以是一个DataFrame

    # 将目标时间序列 y 向后移动一个时间步，第一个值相当于变成0了
    # 这模拟了时间序列的因果关系，即当前时间步的目标值依赖于上一时间步的特征。
    df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
    # 用 0.0 填充缺失值
    df_yshift[target]=df_yshift[target].fillna(0.) # 使用[target]让df_yshift变成一个series，成为一个单一的列，这样填充的时候

    # 将计算出的滞后目标列赋值给输入数据的对应列
    df_x[target] = df_yshift

    '''
    .values方法将 DataFrame 转换为 NumPy 数组，返回一个二维数组（矩阵）
    .astype('float32')将 NumPy 数组的数据类型转换为 float32（32位浮点数），为了确保数据在后续的 PyTorch 计算中使用统一的浮点数精度。
    .transpose()对 NumPy 数组进行转置操作。转置会将数组的行和列互换。在某些模型中，输入数据的维度需要是 (通道数, 时间步数)，而不是 (时间步数, 通道数)
    .from_numpy()将 NumPy 数组转换为 PyTorch 张量（Tensor）
    '''
    data_x = df_x.values.astype('float32').transpose()    
    data_y = df_y.values.astype('float32').transpose()
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    # Variable将 PyTorch 张量包装成具有自动求导功能的对象。这是早期版本的 PyTorch 中常用的做法
    # 但在 PyTorch 0.4.0 及以上版本中，torch.Tensor 已经直接支持自动求导功能
    x, y = Variable(data_x), Variable(data_y)
    return x, y

# 模型训练，训练模型一个 epoch，并返回注意力分数和损失。
def train(epoch, traindata, traintarget, model, optimizer,log_interval,epochs):
    model.train() # 将模型设置为训练模式
    x, y = traindata[0:1], traintarget[0:1] # 获取训练数据和目标的第一个样本
        
    optimizer.zero_grad() # 梯度归零
    epochpercentage = (epoch/float(epochs))*100 # 得到当前的epoch属于总epoch的百分之多少
    output = model(x) # 模型前向传播，得到输出

    attentionscores = model.fs_attention # 得到注意力分数
    
    loss = F.mse_loss(output, y) # 得到损失值
    loss.backward() # 反向传播,得到梯度
    optimizer.step() # 使用优化器更新模型参数

    # 根据日志间隔打印训练信息
    if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
        print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))

    return attentionscores.data, loss # 返回注意力分数和损失值

# 因果原因分析,发现目标时间序列的潜在原因，用PIVM验证这些原因的有效性，并估计原因与目标之间的时延。
def findcauses(target, cuda, epochs, kernel_size, layers, 
               log_interval, lr, optimizername, seed, dilation_c, significance, file):
    '''
    target：要预测的目标变量。
    log_interval：每经过多少个epoch后打印一次训练进度和损失信息。
    lr：学习率
    dilation_c：膨胀因子，允许卷积核在不增加参数量的情况下扩大感受野。
    significance：显著性水平。用于验证潜在原因的有效性。它决定了测试差异与原始差异之间的倍数阈值，超过该阈值的潜在原因将被移除。
    '''
    print("\n", "Analysis started for target: ", target)
    torch.manual_seed(seed) # 固定种子
    
    X_train, Y_train = preparedata(file, target) # 数据准备
    
    '''
    unsqueeze :PyTorch 中用于增加张量维度的操作，适应需要特定形状的操作。它的作用是在指定的位置插入一个大小为 1 的维度。
    假设有一个张量 x，其形状为 [m, n]，如果调用 x.unsqueeze(2)，则会在第 2 个维度（从 0 开始计数）插入一个大小为 1 的维度，结果张量的形状变为 [m, n, 1]。

    contiguous :确保张量内存连续的操作。在 PyTorch 中，张量的内存布局可能不是连续的，这通常发生在张量经过某些操作（如 view、transpose、permute 等）之后。
    如果张量的内存不连续，可能会导致后续操作（如 view）报错。


    .unsqueeze(0).contiguous()：确保在增加维度之后，张量的内存布局是连续的，避免以后报错。
    '''
    X_train = X_train.unsqueeze(0).contiguous()
    Y_train = Y_train.unsqueeze(2).contiguous()

    input_channels = X_train.size()[1]


       
    targetidx = pd.read_csv(file).columns.get_loc(target) # 目标列在csv的列名列表中的索引位置

    # 模型初始化
    model = ADDSTCN(targetidx, input_channels, layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
    if cuda:
        model.cuda()
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

    '''
    getattr 是 Python 的内置函数，用于动态地从模块或对象中获取属性或方法。
    torch.optim 是一个提供各种优化器的模块
    根据字符串 optimizername 动态地选择并初始化一个优化器。这种方式的优点是代码的灵活性很高，可以通过配置文件或用户输入动态选择优化器，而无需硬编码。
    '''
    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)    
    
    scores, firstloss = train(1, X_train, Y_train, model, optimizer,log_interval,epochs) # 在第一个epoch上训练模型，并返回注意力分数和损失。
    '''
    .cpu() 是 PyTorch 中的一个方法，用于将张量从 GPU（如果它在 GPU 上）移动到 CPU 内存。
    张量可以存储在 CPU 或 GPU 上，具体取决于是否启用了 CUDA（GPU 加速）。如果模型或张量在 GPU 上进行计算，其结果也会存储在 GPU 内存中。
    然而，某些操作（如打印、保存到文件或与其他 Python 数据结构交互）需要张量在 CPU 上。
    如果张量已经在 CPU 上，则 .cpu() 不会改变它。

    .data 是 PyTorch 中的一个属性，用于访问张量的底层数据。它返回一个与原张量共享内存的新张量。
    在早期版本的 PyTorch 中，.data 被广泛用于访问张量的原始数据。然而，在 PyTorch 0.4.0 及更高版本中，.data 的使用已经被废弃，因为直接访问底层数据可能会导致一些问题（例如梯度信息丢失）。
    现在，推荐使用 .detach() 来分离张量，同时保留其数据。
    y = x.detach()  # 返回一个不参与梯度计算的新张量

    .item()是 PyTorch 中的一个方法，用于将一个包含单个值的张量转换为 Python 标量。
    .item() 只能用于大小为 1 的张量（例如 torch.tensor([1.0]) 或 torch.tensor(1.0)）
    '''
    firstloss = firstloss.cpu().data.item()

    for ep in range(2, epochs+1):
        scores, realloss = train(ep, X_train, Y_train, model, optimizer,log_interval,epochs)
    realloss = realloss.cpu().data.item()
    
    '''
    view(-1): PyTorch 中用于改变张量形状的方法，类似于 NumPy 中的 reshape。这里view(-1) 是为了将多维张量转换为一维张量。
    detach(): 返回一个新的张量，该张量与原始张量共享数据，但不会参与梯度计算。
    numpy(): PyTorch 中用于将张量转换为 NumPy 数组的方法。
    '''
    s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True) # 对注意力分数进行降序排列
    indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy()) # 获取按降序排列的索引
    
    #通过注意力解释找到tau：区分潜在原因与非因果时间序列的阈值
    if len(s)<=5:
        potentials = []
        for i in indices:
            if scores[i]>1.:
                potentials.append(i)
    else:
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if s[i]<1.: #tau应该大于或等于1，所以只考虑分数>= 1
                break
            gap = s[i]-s[i+1]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)
        
        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if index<((len(s)-1)/2): #gap应在前半部分
                if index>0:
                    ind=index #gap的索引应大于0，除非第二个分数<1
                    break
        if ind<0:
            ind = 0
                
        potentials = indices[:ind+1].tolist()
    # 打印潜在原因
    print("Potential causes: ", potentials)
    # 深拷贝潜在原因列表，用于后续验证
    validated = copy.deepcopy(potentials)
    
    #应用PIVM（置换值）检查潜在原因是否为真实原因
    for idx in potentials:
        random.seed(seed)  # 固定随机种子以确保结果可复现
        X_test2 = X_train.clone().cpu().numpy()  # 克隆训练数据并移回CPU
        random.shuffle(X_test2[:, idx, :][0])  # 置换特定索引的时间序列数据
        shuffled = torch.from_numpy(X_test2)  # 将置换后的数据转换为PyTorch张量
        if cuda:
            shuffled=shuffled.cuda()
        model.eval()  # 将模型设置为评估模式
        output = model(shuffled)  # 前向传播，得到输出
        testloss = F.mse_loss(output, Y_train)  # 计算测试损失
        testloss = testloss.cpu().data.item()  # 将测试损失从GPU移回CPU并转换为标量
        
        diff = firstloss - realloss  # 计算初始损失与最终损失之间的差异
        testdiff = firstloss - testloss  # 计算初始损失与测试损失之间的差异

        # 如果测试差异大于原始差异的显著性倍数，则移除该潜在原因
        if testdiff>(diff*significance): 
            validated.remove(idx) 
    
 
    weights = [] # 存储各层的权重
    
    # 通过解释卷积核权重发现因果关系的时间延迟
    for layer in range(layers):
        weight = model.dwn.network[layer].net[0].weight.abs().view(model.dwn.network[layer].net[0].weight.size()[0], model.dwn.network[layer].net[0].weight.size()[2])
        weights.append(weight)

    causeswithdelay = dict() # 存储因果关系及其对应的时间延迟    
    for v in validated: 
        totaldelay=0    
        for k in range(len(weights)):
            w=weights[k]
            row = w[v]
            twolargest = heapq.nlargest(2, row)
            m = twolargest[0]
            m2 = twolargest[1]
            if m > m2:
                index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
            else:
                # 取第一个滤波器
                index_max=0
            delay = index_max *(dilation_c**k)
            totaldelay+=delay
        if targetidx != v:
            causeswithdelay[(targetidx, v)]=totaldelay
        else:
            causeswithdelay[(targetidx, v)]=totaldelay+1
    print("Validated causes: ", validated)
    
    return validated, causeswithdelay, realloss, scores.view(-1).cpu().detach().numpy().tolist()





