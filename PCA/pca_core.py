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
                uthresh = sigma * np.sqrt(2 * np.log(len(data))) * self.threshold_coeff
                new_coeffs.append(pywt.threshold(coeffs[j], value=uthresh, mode='soft'))
            # 3. 小波重构
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
    def __init__(self, n_components=6, sparsity_k=4, sigma=1.0):
        self.n_components = n_components  # 保留的主成分数量
        self.sparsity_k = sparsity_k      # 稀疏性控制
        self.sigma = sigma                # 高斯核带宽
        self.V_sparse = None              # 稀疏主成分矩阵
        self.pseudo_values = None         # 伪特征值
        self.lmvt = None                  # LMVT阈值
        self.normal_baseline_T2 = None    # 正常工况T²贡献度基准
        self.normal_baseline_SPE = None   # 正常工况SPE贡献度基准

    def _compute_hsic_matrix(self, X):
        """核相关矩阵计算，捕捉特征间的非线性相关关系"""
        n_samples, n_vars = X.shape
        H = np.eye(n_samples) - (1.0 / n_samples) * np.ones((n_samples, n_samples))
        K_list = []
        for i in range(n_vars):
            xi = X[:, i].reshape(-1, 1)
            dist = squareform(pdist(xi, 'sqeuclidean'))
            Ki = np.exp(-dist / (2 * self.sigma**2))
            K_list.append(H @ Ki @ H)
        
        hsic_matrix = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(i, n_vars):
                score = np.trace(K_list[i] @ K_list[j]) / (n_samples - 1)**2
                hsic_matrix[i, j] = hsic_matrix[j, i] = score
        return hsic_matrix

    def _calculate_all_contributions(self, X_scaled):
        """贡献度计算：分解 T2 和 SPE 到各个维度"""
        lambda_inv = np.diag(1.0 / (self.pseudo_values + 1e-10))
        scores = X_scaled @ self.V_sparse
        # T2 贡献度 (基于投影矩阵分解)
        T2_cont_matrix = np.abs(X_scaled * (scores @ lambda_inv @ self.V_sparse.T))
        # SPE 贡献度 (基于残差平方)
        X_hat = scores @ self.V_sparse.T
        SPE_cont_matrix = (X_scaled - X_hat)**2
        return np.mean(T2_cont_matrix, axis=0), np.mean(SPE_cont_matrix, axis=0)

    def fit_offline(self, X_scaled):
        """离线建模逻辑"""
        self.n_vars = X_scaled.shape[1]
        self.lmvt = 1.0 / self.n_vars
        kernel_corr = self._compute_hsic_matrix(X_scaled)
        
        sigma_t = kernel_corr.copy()
        delta_t = np.eye(self.n_vars)
        self.V_sparse = np.zeros((self.n_vars, self.n_components))
        self.pseudo_values = np.zeros(self.n_components)
        
        for t in range(self.n_components):
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
            
            I_qq = np.eye(self.n_vars) - np.outer(q_t, q_t)
            sigma_t = I_qq @ sigma_t @ I_qq
            delta_t = delta_t @ I_qq

        self.normal_baseline_T2, self.normal_baseline_SPE = self._calculate_all_contributions(X_scaled)
        print("离线建模完成。已建立双指标基准。")

    def compute_monitoring_stats(self, X_scaled):
        """线上监控"""
        scores = X_scaled @ self.V_sparse
        lambda_inv = np.diag(1.0 / (self.pseudo_values + 1e-10))
        T2 = np.sum((scores @ lambda_inv) * scores, axis=1)
        X_hat = scores @ self.V_sparse.T
        SPE = np.sum((X_scaled - X_hat)**2, axis=1)
        return T2, SPE

    def trigger_diagnose(self, X_scaled_fault):
        """故障诊断：计算 DCC 并判定根因"""
        fault_cont_T2, fault_cont_SPE = self._calculate_all_contributions(X_scaled_fault)
        DCC = np.abs(fault_cont_T2 - self.normal_baseline_T2)
        DCC_norm = DCC / (np.sum(DCC) + 1e-10)
        root_causes = np.where(DCC_norm > self.lmvt)[0]
        
        return {
            'before_T2': self.normal_baseline_T2,
            'after_T2': fault_cont_T2,
            'before_SPE': self.normal_baseline_SPE,
            'after_SPE': fault_cont_SPE,
            'dcc_norm': DCC_norm,
            'root_causes': root_causes
        }

# ==========================================
# 第三部分：执行与完整绘图
# ==========================================

if __name__ == "__main__":
    # --- 1. 数据准备 ---
    np.random.seed(42)
    n_vars = 14
    df_normal = pd.DataFrame(np.random.normal(0, 0.5, (400, n_vars))) 

    # --- 2. 流程初始化 ---
    denoiser = WaveletDenoiser()
    scaler = RobustScaler()
    model = RNSPCA(n_components=6, sparsity_k=4)

    # --- 3. 离线建模流水线 ---
    X_train_denoised = denoiser.transform(df_normal)
    scaler.fit(X_train_denoised)
    X_train_scaled = scaler.transform(X_train_denoised)
    model.fit_offline(X_train_scaled)

    # --- 4. 在线监控模拟 ---
    test_data = np.random.normal(0, 0.5, (200, n_vars))
    test_data[100:, 9] = (test_data[100:, 8]**2) * 3.0 + 2.0 
    df_test = pd.DataFrame(test_data)

    X_test_denoised = denoiser.transform(df_test)
    X_test_scaled = scaler.transform(X_test_denoised)
    T2_vals, SPE_vals = model.compute_monitoring_stats(X_test_scaled)

    # --- 5. 触发诊断 ---
    X_fault_segment = X_test_scaled[100:130]
    diag_res = model.trigger_diagnose(X_fault_segment)

    # --- 结果可视化 ---
    plt.figure(figsize=(10, 16))
    x_idx = np.arange(1, n_vars + 1)
    bar_width = 0.35

    # 1. 系统整体 T2 Chart
    plt.subplot(5, 1, 1)
    plt.plot(T2_vals, color='blue')
    plt.axvline(x=100, color='red', linestyle='--')
    plt.title('Online Monitoring: $T^2$')

    # 2. 系统整体 SPE Chart
    plt.subplot(5, 1, 2)
    plt.plot(SPE_vals, color='green')
    plt.axvline(x=100, color='red', linestyle='--')
    plt.title('Online Monitoring: $SPE$')

    # 3. T2 贡献度前后对比图
    plt.subplot(5, 1, 3)
    plt.bar(x_idx - bar_width/2, diag_res['before_T2'], bar_width, label='Normal (Before)', color='gray')
    plt.bar(x_idx + bar_width/2, diag_res['after_T2'], bar_width, label='Fault (After)', color='blue')
    plt.title('$T^2$ Contribution Comparison')
    plt.xticks(x_idx)
    plt.legend()

    # 4. SPE 贡献度前后对比图 
    plt.subplot(5, 1, 4)
    plt.bar(x_idx - bar_width/2, diag_res['before_SPE'], bar_width, label='Normal (Before)', color='gray')
    plt.bar(x_idx + bar_width/2, diag_res['after_SPE'], bar_width, label='Fault (After)', color='green')
    plt.title('$SPE$ Contribution Comparison')
    plt.xticks(x_idx)
    plt.legend()

    # 5. DCC-LMVT 诊断结果 
    plt.subplot(5, 1, 5)
    plt.bar(x_idx, diag_res['dcc_norm'], color='orange')
    plt.axhline(y=model.lmvt, color='red', label=f'LMVT ({model.lmvt:.3f})')
    plt.title('Root Cause Diagnosis (DCC-LMVT)')
    plt.xticks(x_idx)
    plt.legend()

    plt.tight_layout()
    plt.savefig('diagnostic_report.png')

    print(f"检测到的根因索引: {diag_res['root_causes']}")