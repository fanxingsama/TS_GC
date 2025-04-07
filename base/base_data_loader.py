import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        '''
        dataset：要加载的数据集。
        batch_size：每个批次的样本数量。
        shuffle：是否随机打乱数据。
        validation_split：验证集的比例或数量。
        num_workers：加载数据时使用的子进程数量。
        collate_fn：用于将多个样本组合成一个批次的函数，默认为 default_collate
        '''
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    # 划分数据集
    def _split_sampler(self, split):
        if split == 0.0: # 如果 split 为 0.0，表示不划分验证集，返回 None
            return None, None

        idx_full = np.arange(self.n_samples) # 生成一个包含所有样本索引的数组 idx_full。

        np.random.seed(0)
        np.random.shuffle(idx_full)

        # 根据split的类型，划分数据集
        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        # 划分验证集和训练集的索引
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # 使用 SubsetRandomSampler 创建训练集和验证集的采样器
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # 关闭 shuffle 选项，因为采样器已经提供了随机性
        self.shuffle = False
        self.n_samples = len(train_idx)

        # 返回训练集和验证集的采样器
        return train_sampler, valid_sampler

    def split_validation(self):
        # 如果没有验证集采样器，返回None
        if self.valid_sampler is None:
            return None
        else:
            # 使用验证集采样器创建一个新的 DataLoader 实例，返回验证集的数据加载器。
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
