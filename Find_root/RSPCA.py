import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pywt

# ==========================================
# 第一部分：独立预处理器
# ==========================================

class WaveletDenoiser:
    """
    增强版小波去噪器
    wavelet: 小波基名称 (推荐 'db8' 或 'sym8' 处理平滑信号)
    level: 分解层数 (越高去噪越狠, 建议 3-5)
    threshold_coeff: 阈值系数 (越大去噪越狠, 1.0 为标准值)
    """
    def __init__(self, wavelet='sym8', level=3, threshold_coeff=2.0):
        self.wavelet = wavelet
        self.level = level
        self.threshold_coeff = threshold_coeff

    def transform(self, X):
        data = X.values if isinstance(X, pd.DataFrame) else X
        denoised_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            # 1. 小波分解
            coeffs = pywt.wavedec(data[:, i], self.wavelet, level=self.level)
            # 2. 对每一层高频系数进行阈值处理
            new_coeffs = [coeffs[0]] # 保留低频近似部分
            for j in range(1, len(coeffs)):
                sigma = np.median(np.abs(coeffs[j])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(data))) * self.threshold_coeff # 计算自适应阈值
                new_coeffs.append(pywt.threshold(coeffs[j], value=uthresh, mode='soft')) # 软阈值处理
            # 3. 小波重构，用处理后的系数重建信号
            res = pywt.waverec(new_coeffs, self.wavelet)
            denoised_data[:, i] = res[:len(data)]
        return denoised_data

class RobustScaler:
    """鲁棒标准化器：使用中位数和MAD，避免工业异常值干扰"""
    def __init__(self):
        self.median = None                # 存储训练数据的中位数
        self.mad = None                   # 存储训练数据的绝对中位差

    def fit(self, X):
        """训练阶段：计算并保存中位数和MAD"""
        self.median = np.median(X, axis=0)
        self.mad = np.median(np.abs(X - self.median), axis=0)
        self.mad[self.mad == 0] = 1e-6    # 避免除0错误

    def transform(self, X):
        """标准化公式：(X - 中位数) / (1.4826 * MAD)"""
        return (X - self.median) / (1.4826 * self.mad)

# ==========================================
# 第二部分：核心模型
# ==========================================

class RNSPCA:
    """稀疏核主成分分析（RNSPCA）"""
    def __init__(self, n_components=6, sparsity_k=4, sigma=1.0, window_size=3):
        self.n_components = n_components  # 保留的主成分数量
        self.sparsity_k = sparsity_k      # 稀疏性控制
        self.sigma = sigma                # 高斯核带宽
        self.window_size = window_size  # 新增：窗口大小
        self.V_sparse = None              # 稀疏主成分矩阵
        self.pseudo_values = None         # 伪特征值
        # self.lmvt = None                  # LMVT阈值
        self.normal_baseline_T2 = None    # 正常工况T²贡献度基准
        self.normal_baseline_SPE = None   # 正常工况SPE贡献度基准
        self.n_original_vars = None # 记录原始变量数

    # 时序嵌入
    def _create_lagged_matrix(self, X):
        """
        输入 shape: (samples, features)
        输出 shape: (samples - window + 1, features * window)
        """
        n_samples, n_features = X.shape # n_samples：时间序列长度，n_features：变量数
        if n_samples < self.window_size:
            raise ValueError(f"数据样本数 ({n_samples}) 小于窗口大小 ({self.window_size})")
        
        # 这种方式将 t, t-1, t-2... 拼接到一行
        # 结果列顺序: [Var1_t, Var2_t, ..., Var1_t-1, Var2_t-1, ...]
        new_features = []
        for i in range(self.window_size):
            # 向后截取，模拟当前时刻包含过去信息
            # start: 窗口大小-1-i, end: 样本总数-i
            start = self.window_size - 1 - i
            end = n_samples - i
            chunk = X[start:end, :]
            new_features.append(chunk)
            
        # 水平拼接
        X_lagged = np.hstack(new_features)
        return X_lagged
    
    # HSIC矩阵计算，捕捉特征间的非线性相关关系
    def _compute_hsic_matrix(self, X):
        n_samples, n_vars = X.shape
        K_centered_list = []
        
        for i in range(n_vars):
            xi = X[:, i].reshape(-1, 1)
            dist_sq = squareform(pdist(xi, 'sqeuclidean'))
            K = np.exp(-dist_sq / (2 * self.sigma**2))
            k_mean_row = K.mean(axis=0, keepdims=True)
            k_mean_col = K.mean(axis=1, keepdims=True)
            k_mean_all = K.mean()
            K_centered = K - k_mean_row - k_mean_col + k_mean_all
            
            K_centered_list.append(K_centered)
        
        hsic_matrix = np.zeros((n_vars, n_vars))
        # 归一化系数
        normalization = 1.0 / ((n_samples - 1)**2)
        
        for i in range(n_vars):
            print(f"正在计算HSIC矩阵: 协方差矩阵 {i+1}/{n_vars}...")
            hsic_matrix[i, i] = np.sum(K_centered_list[i] * K_centered_list[i]) * normalization
            for j in range(i + 1, n_vars):
                score = np.sum(K_centered_list[i] * K_centered_list[j]) * normalization
                hsic_matrix[i, j] = score
                hsic_matrix[j, i] = score
                
        print("\nHSIC 矩阵计算完成。")
        return hsic_matrix
    
    # 贡献度计算
    def _calculate_all_contributions(self, X_scaled):
        """贡献度计算：分解 T2 和 SPE 到各个维度"""
        lambda_inv = np.diag(1.0 / (self.pseudo_values + 1e-10))
        scores = X_scaled @ self.V_sparse
        # T2 贡献度 (基于投影矩阵分解)
        T2_cont_matrix = np.abs(X_scaled * (scores @ lambda_inv @ self.V_sparse.T))
        # SPE 贡献度 (基于残差平方)
        X_hat = scores @ self.V_sparse.T
        SPE_cont_matrix = (X_scaled - X_hat)**2
        
        # 取平均得到 (n_vars * window) 长度的向量
        mean_T2_ext = np.mean(T2_cont_matrix, axis=0)
        mean_SPE_ext = np.mean(SPE_cont_matrix, axis=0)
        
        # 2. 聚合回原始维度 (Aggregation)
        # 扩展列结构: [Var1_t0, Var2_t0, ..., Var1_t1, Var2_t1, ...]
        # 我们需要把所有属于 Var1 的贡献加起来
        final_T2 = np.zeros(self.n_original_vars)
        final_SPE = np.zeros(self.n_original_vars)
        
        for w in range(self.window_size):
            start_idx = w * self.n_original_vars
            end_idx = start_idx + self.n_original_vars
            final_T2 += mean_T2_ext[start_idx:end_idx]
            final_SPE += mean_SPE_ext[start_idx:end_idx]
            
        # 可以选择取平均或者求和，求和能放大信号
        return final_T2, final_SPE

    # 离线建模
    def fit_offline(self, X_scaled):
        self.n_original_vars = X_scaled.shape[1]
        # self.lmvt = 1.0 / self.n_original_vars
        
        # 1. 应用滑动窗口
        print(f"应用滑动窗口 (size={self.window_size})...")
        X_lagged = self._create_lagged_matrix(X_scaled)
        n_lagged_vars = X_lagged.shape[1]
        
        kernel_corr = self._compute_hsic_matrix(X_lagged)
    
        # 初始化特征值分解相关变量
        sigma_t = kernel_corr.copy()  # 待分解的核矩阵副本
        delta_t = np.eye(n_lagged_vars)  # 单位矩阵，用于正交化更新
        self.V_sparse = np.zeros((n_lagged_vars, self.n_components))  # 稀疏特征向量矩阵
        self.pseudo_values = np.zeros(self.n_components)  # 伪特征值（稀疏主成分的方差解释量）
        
        # 3. 迭代提取稀疏正交主成分（稀疏PCA变体）
        for t in range(self.n_components):
            print(f"正在提取第 {t+1}/{self.n_components} 个主成分...")
            # 对核矩阵进行特征值分解（求解最大特征值对应的特征向量）
            eig_vals, eig_vecs = np.linalg.eigh(sigma_t)
            idx = np.argsort(eig_vals)[-1]  # 取最大特征值的索引
            v_t = eig_vecs[:, idx]
            
            # 稀疏化：仅保留绝对值最大的k个元素（其余置0）
            v_sparse = np.zeros_like(v_t)
            top_k_idx = np.argsort(np.abs(v_t))[-self.sparsity_k:]
            v_sparse[top_k_idx] = v_t[top_k_idx]
            v_sparse /= (np.linalg.norm(v_sparse) + 1e-10)  # 归一化（避免除零）
            
            # 正交化更新：剔除已提取的主成分，避免重复
            q_t = delta_t @ v_sparse
            self.V_sparse[:, t] = v_sparse  # 保存稀疏特征向量
            self.pseudo_values[t] = v_sparse.T @ sigma_t @ v_sparse  # 保存伪特征值
            
            # 更新核矩阵和正交化矩阵
            I_qq = np.eye(n_lagged_vars) - np.outer(q_t, q_t)  # 正交投影矩阵
            sigma_t = I_qq @ sigma_t @ I_qq  # 更新核矩阵
            delta_t = delta_t @ I_qq  # 更新正交化矩阵

        # 4. 计算T2和SPE基准值
        self.normal_baseline_T2, self.normal_baseline_SPE = self._calculate_all_contributions(X_lagged)
        print("离线建模完成。已建立双指标基准。")

    # 故障诊断
    def trigger_diagnose(self, X_scaled_fault):
        # 1. 对故障数据应用同样的滑动窗口
        X_fault_lagged = self._create_lagged_matrix(X_scaled_fault)
        fault_cont_T2, fault_cont_SPE = self._calculate_all_contributions(X_fault_lagged)
        DCC = np.abs(fault_cont_T2 - self.normal_baseline_T2)
        # DCC = np.abs(fault_cont_SPE - self.normal_baseline_SPE)
        DCC_norm = DCC / (np.sum(DCC) + 1e-10)
        # root_causes = np.where(DCC_norm > self.lmvt)[0]
        
        return {
            'before_T2': self.normal_baseline_T2,
            'after_T2': fault_cont_T2,
            'before_SPE': self.normal_baseline_SPE,
            'after_SPE': fault_cont_SPE,
            'dcc_norm': DCC_norm,
            # 'root_causes': root_causes
        }