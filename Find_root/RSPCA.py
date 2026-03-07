import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pywt

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
    def __init__(self):
        self.median = None                
        self.mad = None                   

    def fit(self, X):
        self.median = np.median(X, axis=0)
        self.mad = np.median(np.abs(X - self.median), axis=0)
        self.mad[self.mad < 1e-4] = 1.0 / 1.4826

    def transform(self, X):
        return (X - self.median) / (1.4826 * self.mad)


class RNSPCA:
    """稀疏核主成分分析（RNSPCA）- 仅聚焦 SPE 统计量"""
    def __init__(self, n_components=6, sparsity_k=4, sigma=1.0, window_size=3, use_window=True, alpha=0.01):
        self.n_components = n_components 
        self.sparsity_k = sparsity_k 
        self.sigma = sigma * 1.4826 
        self.window_size = window_size 
        self.use_window = use_window      
        self.alpha = alpha        # 设定显著性水平 (如 0.01 对应 99% 分位数)
        self.V_sparse = None
        self.pseudo_values = None
        self.normal_baseline_SPE = None
        self.n_original_vars = None
        self.SPE_threshold = None         

    def _create_lagged_matrix(self, X):
        if not self.use_window or self.window_size <= 1:
            return X  
            
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
    
    def _calculate_spe_contributions(self, X_scaled):
        scores = X_scaled @ self.V_sparse
        X_hat = scores @ self.V_sparse.T
        SPE_cont_matrix = (X_scaled - X_hat)**2
        
        mean_SPE_ext = np.mean(SPE_cont_matrix, axis=0)
        final_SPE = np.zeros(self.n_original_vars)
        
        if self.use_window and self.window_size > 1:
            for w in range(self.window_size):
                start_idx = w * self.n_original_vars
                end_idx = start_idx + self.n_original_vars
                final_SPE += mean_SPE_ext[start_idx:end_idx]
        else:
            final_SPE = mean_SPE_ext
            
        return final_SPE

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

        self.normal_baseline_SPE = self._calculate_spe_contributions(X_lagged)
        
        # ============== 新增: 计算并保存每个变量专属的 SPE 阈值 ==============
        scores_normal = X_lagged @ self.V_sparse
        X_hat_normal = scores_normal @ self.V_sparse.T
        SPE_cont_matrix_normal = (X_lagged - X_hat_normal)**2
        
        var_SPE_series_normal = np.zeros((SPE_cont_matrix_normal.shape[0], self.n_original_vars))
        if self.use_window and self.window_size > 1:
            for w in range(self.window_size):
                start_idx = w * self.n_original_vars
                end_idx = start_idx + self.n_original_vars
                var_SPE_series_normal += SPE_cont_matrix_normal[:, start_idx:end_idx]
        else:
            var_SPE_series_normal = SPE_cont_matrix_normal
            
        # 计算每个变量的阈值 (按设定的显著性水平 alpha，如 99% 或 99.9% 分位数)
        self.var_SPE_thresholds = np.percentile(var_SPE_series_normal, (1 - self.alpha) * 100, axis=0)
        # =====================================================================
        
        # 计算 SPE 阈值
        scores_normal = X_lagged @ self.V_sparse
        X_hat_normal = scores_normal @ self.V_sparse.T
        SPE_scores_normal = np.sum((X_lagged - X_hat_normal)**2, axis=1)
        self.SPE_threshold = np.percentile(SPE_scores_normal, (1 - self.alpha) * 100)
        
        print(f"离线建模完成。全局异常阈值设为 -> SPE: {self.SPE_threshold:.4f}")

    def predict_global_anomaly(self, X_scaled_fault):
        """仅返回 SPE 统计量"""
        X_fault_lagged = self._create_lagged_matrix(X_scaled_fault)
        scores = X_fault_lagged @ self.V_sparse
        
        X_hat = scores @ self.V_sparse.T
        stat_scores = np.sum((X_fault_lagged - X_hat)**2, axis=1)
        threshold = self.SPE_threshold
        
        if self.use_window and self.window_size > 1:
            pad_length = self.window_size - 1
            padded_stat = np.pad(stat_scores, (pad_length, 0), constant_values=np.nan)
        else:
            padded_stat = stat_scores
            
        return padded_stat, threshold

    def trigger_diagnose(self, X_scaled_fault):
        X_fault_lagged = self._create_lagged_matrix(X_scaled_fault)
        fault_cont_SPE = self._calculate_spe_contributions(X_fault_lagged)
        
        DCC = np.abs(fault_cont_SPE - self.normal_baseline_SPE)
        DCC_norm = DCC / (np.sum(DCC) + 1e-10)
        
        return {
            'before_SPE': self.normal_baseline_SPE,
            'after_SPE': fault_cont_SPE,
            'dcc_norm': DCC_norm,
        }
        
    def get_variable_spe_series(self, X_scaled_fault):
        """获取每个变量随时间变化的 SPE 异常分数序列"""
        X_fault_lagged = self._create_lagged_matrix(X_scaled_fault)
        scores = X_fault_lagged @ self.V_sparse
        X_hat = scores @ self.V_sparse.T
        SPE_cont_matrix = (X_fault_lagged - X_hat)**2
        
        # 将滑动窗口中的多列对应回原始变量并求和
        var_SPE_series = np.zeros((SPE_cont_matrix.shape[0], self.n_original_vars))
        if self.use_window and self.window_size > 1:
            for w in range(self.window_size):
                start_idx = w * self.n_original_vars
                end_idx = start_idx + self.n_original_vars
                var_SPE_series += SPE_cont_matrix[:, start_idx:end_idx]
        else:
            var_SPE_series = SPE_cont_matrix
            
        # 填充由于滑动窗口导致的前几个缺失时间点 (保证时间长度与原始测试集一致)
        if self.use_window and self.window_size > 1:
            pad_length = self.window_size - 1
            padded_series = np.pad(var_SPE_series, ((pad_length, 0), (0, 0)), constant_values=np.nan)
            return padded_series
            
        return var_SPE_series