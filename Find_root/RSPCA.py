import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pywt
from scipy import stats

# ==========================================
# 第一部分：独立预处理器 (保持不变)
# ==========================================

class WaveletDenoiser:
    def __init__(self, wavelet='sym8', level=3, threshold_coeff=2.0):
        self.wavelet = wavelet
        self.level = level
        self.threshold_coeff = threshold_coeff

    def transform(self, X):
        data = X.values if isinstance(X, pd.DataFrame) else X
        denoised_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            coeffs = pywt.wavedec(data[:, i], self.wavelet, level=self.level)
            new_coeffs = [coeffs[0]]
            for j in range(1, len(coeffs)):
                sigma = np.median(np.abs(coeffs[j])) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(data))) * self.threshold_coeff
                new_coeffs.append(pywt.threshold(coeffs[j], value=uthresh, mode='soft'))
            res = pywt.waverec(new_coeffs, self.wavelet)
            denoised_data[:, i] = res[:len(data)]
        return denoised_data

class RobustScaler:
    """鲁棒标准化器：修复极小方差变量导致的数值爆炸问题"""
    def __init__(self):
        self.median = None                
        self.mad = None                   

    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.mad = np.median(np.abs(X - self.median), axis=0)
        
        # 【关键修改】：如果 MAD 非常小（小于 1e-4），说明该变量几乎恒定。
        # 我们不要用极小数去除它，而是强制让它除以 1.0 (通过反除 1.4826)，保持原本的微小波动即可。
        self.mad[self.mad < 1e-4] = 1.0 / 1.4826

    def transform(self, X):
        """标准化公式：(X - 中位数) / (1.4826 * MAD)"""
        return (X - self.median) / (1.4826 * self.mad)

# ==========================================
# 第二部分：核心模型 (已更新)
# ==========================================

class RNSPCA:
    """稀疏核主成分分析（RNSPCA）- 支持自定义统计量和滑动窗口"""
    def __init__(self, n_components=6, sparsity_k=4, sigma=1.0, window_size=3, use_window=True, alpha=0.01):
        self.n_components = n_components # 主成分数量
        self.sparsity_k = sparsity_k # 每个主成分保留的非零元素数量
        self.sigma = sigma * 1.4826 # 核宽度参数
        self.window_size = window_size # 滑动窗口大小 (默认为3，表示当前时刻和前两个时刻的联合分析)
        self.use_window = use_window      # 新增：滑动窗口开关 
        self.alpha = alpha             # 显著性水平 (用于计算全局异常阈值)
        self.V_sparse = None
        self.pseudo_values = None
        self.normal_baseline_T2 = None
        self.normal_baseline_SPE = None
        self.n_original_vars = None
        self.S2_threshold = None
        self.SPE_threshold = None         # 新增：SPE 全局异常阈值

    # 时序嵌入
    def _create_lagged_matrix(self, X):
        if not self.use_window or self.window_size <= 1:
            return X  # 不使用滑动窗口时直接返回原数据
            
        n_samples, n_features = X.shape 
        if n_samples < self.window_size:
            raise ValueError(f"数据样本数 ({n_samples}) 小于窗口大小 ({self.window_size})")
        
        new_features = []
        for i in range(self.window_size):
            start = self.window_size - 1 - i
            end = n_samples - i
            chunk = X[start:end, :]
            new_features.append(chunk)
            
        X_lagged = np.hstack(new_features)
        return X_lagged
    
    # HSIC矩阵计算
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
        normalization = 1.0 / ((n_samples - 1)**2)
        
        for i in range(n_vars):
            print(f"正在计算HSIC矩阵: 协方差矩阵 {i+1}/{n_vars}...", end='\r')
            hsic_matrix[i, i] = np.sum(K_centered_list[i] * K_centered_list[i]) * normalization
            for j in range(i + 1, n_vars):
                score = np.sum(K_centered_list[i] * K_centered_list[j]) * normalization
                hsic_matrix[i, j] = score
                hsic_matrix[j, i] = score
                
        print("\nHSIC 矩阵计算完成。")
        return hsic_matrix
    
    # 贡献度计算
    def _calculate_all_contributions(self, X_scaled):
        lambda_inv = np.diag(1.0 / (self.pseudo_values + 1e-10))
        scores = X_scaled @ self.V_sparse
        T2_cont_matrix = np.abs(X_scaled * (scores @ lambda_inv @ self.V_sparse.T))
        
        X_hat = scores @ self.V_sparse.T
        SPE_cont_matrix = (X_scaled - X_hat)**2
        
        mean_T2_ext = np.mean(T2_cont_matrix, axis=0)
        mean_SPE_ext = np.mean(SPE_cont_matrix, axis=0)
        
        final_T2 = np.zeros(self.n_original_vars)
        final_SPE = np.zeros(self.n_original_vars)
        
        # 兼容无滑动窗口的情况
        if self.use_window and self.window_size > 1:
            for w in range(self.window_size):
                start_idx = w * self.n_original_vars
                end_idx = start_idx + self.n_original_vars
                final_T2 += mean_T2_ext[start_idx:end_idx]
                final_SPE += mean_SPE_ext[start_idx:end_idx]
        else:
            final_T2 = mean_T2_ext
            final_SPE = mean_SPE_ext
            
        return final_T2, final_SPE

    # 离线建模
    def fit_offline(self, X_scaled):
        self.n_original_vars = X_scaled.shape[1]
        
        mode_str = f"滑动窗口 (size={self.window_size})" if self.use_window else "禁用滑动窗口"
        print(f"当前模式: {mode_str}...")
        X_lagged = self._create_lagged_matrix(X_scaled)
        n_lagged_vars = X_lagged.shape[1]
        
        kernel_corr = self._compute_hsic_matrix(X_lagged)
    
        sigma_t = kernel_corr.copy()  
        delta_t = np.eye(n_lagged_vars)  
        self.V_sparse = np.zeros((n_lagged_vars, self.n_components))  
        self.pseudo_values = np.zeros(self.n_components)  
        
        for t in range(self.n_components):
            print(f"正在提取第 {t+1}/{self.n_components} 个主成分...")
            eig_vals, eig_vecs = np.linalg.eigh(sigma_t)
            idx = np.argsort(eig_vals)[-1]  
            v_t = eig_vecs[:, idx]
            
            v_sparse = np.zeros_like(v_t)
            top_k_idx = np.argsort(np.abs(v_t))[-self.sparsity_k:]
            v_sparse[top_k_idx] = v_t[top_k_idx]
            v_sparse /= (np.linalg.norm(v_sparse) + 1e-10)  
            
            q_t = delta_t @ v_sparse
            self.V_sparse[:, t] = v_sparse  
            self.pseudo_values[t] = v_sparse.T @ sigma_t @ v_sparse  
            
            I_qq = np.eye(n_lagged_vars) - np.outer(q_t, q_t)  
            sigma_t = I_qq @ sigma_t @ I_qq  
            delta_t = delta_t @ I_qq  

        self.normal_baseline_T2, self.normal_baseline_SPE = self._calculate_all_contributions(X_lagged)
        
        # 1. 计算 T2 (S^2) 阈值 (F分布)
        n = X_lagged.shape[0]
        K = self.n_components
        F_val = stats.f.ppf(1 - self.alpha, K, n - K)
        self.S2_threshold = (K * (n**2 - 1)) / (n * (n - K)) * F_val
        
        # 2. 计算 SPE 阈值 (基于正常样本的经验百分位)
        scores_normal = X_lagged @ self.V_sparse
        X_hat_normal = scores_normal @ self.V_sparse.T
        SPE_scores_normal = np.sum((X_lagged - X_hat_normal)**2, axis=1)
        self.SPE_threshold = np.percentile(SPE_scores_normal, (1 - self.alpha) * 100)
        
        print(f"离线建模完成。已建立双指标基准。")
        print(f"系统全局异常阈值设为 -> T2(S^2): {self.S2_threshold:.4f} | SPE: {self.SPE_threshold:.4f}")

    # 系统全局异常监测计算
    def predict_global_anomaly(self, X_scaled_fault, stat_type='T2'):
        """计算系统随时间变化的全局异常监测统计量 S^2 或 SPE"""
        X_fault_lagged = self._create_lagged_matrix(X_scaled_fault)
        scores = X_fault_lagged @ self.V_sparse
        
        if stat_type == 'T2':
            lambda_inv = np.diag(1.0 / (self.pseudo_values + 1e-10))
            stat_scores = np.sum((scores @ lambda_inv) * scores, axis=1)
            threshold = self.S2_threshold
        elif stat_type == 'SPE':
            X_hat = scores @ self.V_sparse.T
            stat_scores = np.sum((X_fault_lagged - X_hat)**2, axis=1)
            threshold = self.SPE_threshold
        else:
            raise ValueError("stat_type 参数必须是 'T2' 或 'SPE'")
        
        # 处理滑动窗口导致的时间轴对齐问题
        if self.use_window and self.window_size > 1:
            pad_length = self.window_size - 1
            padded_stat = np.pad(stat_scores, (pad_length, 0), constant_values=np.nan)
        else:
            padded_stat = stat_scores
            
        return padded_stat, threshold

    # 故障诊断 (针对根因变量提取)
    def trigger_diagnose(self, X_scaled_fault, stat_type='T2'):
        X_fault_lagged = self._create_lagged_matrix(X_scaled_fault)
        fault_cont_T2, fault_cont_SPE = self._calculate_all_contributions(X_fault_lagged)
        
        if stat_type == 'T2':
            DCC = np.abs(fault_cont_T2 - self.normal_baseline_T2)
        elif stat_type == 'SPE':
            DCC = np.abs(fault_cont_SPE - self.normal_baseline_SPE)
        else:
            raise ValueError("stat_type 参数必须是 'T2' 或 'SPE'")
            
        DCC_norm = DCC / (np.sum(DCC) + 1e-10)
        
        return {
            'before_T2': self.normal_baseline_T2,
            'after_T2': fault_cont_T2,
            'before_SPE': self.normal_baseline_SPE,
            'after_SPE': fault_cont_SPE,
            'dcc_norm': DCC_norm,
        }