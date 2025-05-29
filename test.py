import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

class CausalLagAnalyzer:
    """因果滞后分析器"""
    
    def __init__(self, model, series_names=None, threshold=0.01):
        """
        初始化因果滞后分析器
        
        Args:
            model: 训练好的MutiTS_GC模型
            series_names: 序列名称列表，如果为None则使用数字编号
            threshold: 权重阈值，低于此值的权重将被忽略
        """
        self.model = model
        self.series_num = model.series_num
        self.kernel_size = model.networks[0].first_conv.kernel_size[0]
        self.threshold = threshold
        
        if series_names is None:
            self.series_names = [f'Series_{i}' for i in range(self.series_num)]
        else:
            self.series_names = series_names
            
        # 获取所有网络的第一层权重
        self.weights = self.model.get_first_layer_weights()
    
    def extract_causal_lags(self, method='max_weight', top_k=3):
        """
        提取因果滞后关系
        
        Args:
            method: 滞后提取方法
                - 'max_weight': 使用最大权重位置
                - 'weighted_avg': 使用加权平均位置
                - 'top_k_avg': 使用前k个最大权重的平均位置
            top_k: 当method='top_k_avg'时使用的top-k值
            
        Returns:
            lag_matrix: [series_num, series_num] 的滞后矩阵
            weight_importance: [series_num, series_num] 的权重重要性矩阵
        """
        lag_matrix = np.zeros((self.series_num, self.series_num))
        weight_importance = np.zeros((self.series_num, self.series_num))
        
        for target_idx in range(self.series_num):
            # 获取目标序列对应网络的第一层权重
            # weights[target_idx]: [feature_dim, series_num, kernel_size]
            weight = self.weights[target_idx].detach().cpu().numpy()
            
            # 对每个输入序列（因果源）进行分析
            for source_idx in range(self.series_num):
                # 提取从source_idx到target_idx的权重
                # source_weights: [feature_dim, kernel_size]
                source_weights = weight[:, source_idx, :]
                
                # 计算每个时间位置的重要性（跨特征维度求和）
                time_importance = np.sum(np.abs(source_weights), axis=0)
                weight_importance[target_idx, source_idx] = np.sum(time_importance)
                
                # 根据不同方法计算滞后
                if method == 'max_weight':
                    lag_matrix[target_idx, source_idx] = np.argmax(time_importance)
                
                elif method == 'weighted_avg':
                    if np.sum(time_importance) > self.threshold:
                        weights_norm = time_importance / np.sum(time_importance)
                        lag_matrix[target_idx, source_idx] = np.sum(
                            weights_norm * np.arange(self.kernel_size)
                        )
                    else:
                        lag_matrix[target_idx, source_idx] = 0
                
                elif method == 'top_k_avg':
                    # 找到前k个最大权重的位置
                    top_k_indices = np.argsort(time_importance)[-top_k:]
                    top_k_weights = time_importance[top_k_indices]
                    
                    if np.sum(top_k_weights) > self.threshold:
                        weights_norm = top_k_weights / np.sum(top_k_weights)
                        lag_matrix[target_idx, source_idx] = np.sum(
                            weights_norm * top_k_indices
                        )
                    else:
                        lag_matrix[target_idx, source_idx] = 0
        
        return lag_matrix, weight_importance
    
    def get_significant_causal_relationships(self, weight_threshold=None, lag_matrix=None, weight_importance=None):
        """
        获取显著的因果关系
        
        Args:
            weight_threshold: 权重阈值，如果为None则自动计算
            lag_matrix: 滞后矩阵
            weight_importance: 权重重要性矩阵
            
        Returns:
            causal_relationships: 显著因果关系的列表
        """
        if lag_matrix is None or weight_importance is None:
            lag_matrix, weight_importance = self.extract_causal_lags()
        
        if weight_threshold is None:
            # 使用权重重要性的75分位数作为阈值
            weight_threshold = np.percentile(weight_importance, 75)
        
        causal_relationships = []
        
        for target_idx in range(self.series_num):
            for source_idx in range(self.series_num):
                if source_idx != target_idx:  # 排除自循环
                    weight = weight_importance[target_idx, source_idx]
                    lag = lag_matrix[target_idx, source_idx]
                    
                    if weight > weight_threshold:
                        causal_relationships.append({
                            'source': self.series_names[source_idx],
                            'target': self.series_names[target_idx],
                            'source_idx': source_idx,
                            'target_idx': target_idx,
                            'lag': int(lag),
                            'weight_importance': weight
                        })
        
        # 按权重重要性排序
        causal_relationships.sort(key=lambda x: x['weight_importance'], reverse=True)
        
        return causal_relationships
    
    def plot_lag_heatmap(self, lag_matrix=None, weight_importance=None, 
                        figsize=(12, 10), save_path=None):
        """绘制因果滞后热力图"""
        if lag_matrix is None or weight_importance is None:
            lag_matrix, weight_importance = self.extract_causal_lags()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 滞后热力图
        im1 = ax1.imshow(lag_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('因果滞后矩阵', fontsize=14, fontweight='bold')
        ax1.set_xlabel('源序列 (Cause)', fontsize=12)
        ax1.set_ylabel('目标序列 (Effect)', fontsize=12)
        ax1.set_xticks(range(self.series_num))
        ax1.set_yticks(range(self.series_num))
        ax1.set_xticklabels(self.series_names, rotation=45)
        ax1.set_yticklabels(self.series_names)
        
        # 添加数值标注
        for i in range(self.series_num):
            for j in range(self.series_num):
                text = ax1.text(j, i, f'{lag_matrix[i, j]:.1f}',
                              ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='滞后步数')
        
        # 权重重要性热力图
        im2 = ax2.imshow(weight_importance, cmap='Reds', aspect='auto')
        ax2.set_title('权重重要性矩阵', fontsize=14, fontweight='bold')
        ax2.set_xlabel('源序列 (Cause)', fontsize=12)
        ax2.set_ylabel('目标序列 (Effect)', fontsize=12)
        ax2.set_xticks(range(self.series_num))
        ax2.set_yticks(range(self.series_num))
        ax2.set_xticklabels(self.series_names, rotation=45)
        ax2.set_yticklabels(self.series_names)
        
        # 添加数值标注
        for i in range(self.series_num):
            for j in range(self.series_num):
                text = ax2.text(j, i, f'{weight_importance[i, j]:.2f}',
                              ha="center", va="center", 
                              color="white" if weight_importance[i, j] > np.max(weight_importance)/2 else "black")
        
        plt.colorbar(im2, ax=ax2, label='权重重要性')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weight_distribution(self, target_idx, source_idx, save_path=None):
        """
        绘制特定因果关系的权重分布
        
        Args:
            target_idx: 目标序列索引
            source_idx: 源序列索引
        """
        weight = self.weights[target_idx].detach().cpu().numpy()
        source_weights = weight[:, source_idx, :]  # [feature_dim, kernel_size]
        
        # 计算每个时间位置的重要性
        time_importance = np.sum(np.abs(source_weights), axis=0)
        
        plt.figure(figsize=(10, 6))
        
        # 绘制权重分布
        plt.subplot(2, 1, 1)
        plt.bar(range(self.kernel_size), time_importance, alpha=0.7, color='steelblue')
        plt.title(f'因果权重分布: {self.series_names[source_idx]} → {self.series_names[target_idx]}')
        plt.xlabel('时间滞后位置')
        plt.ylabel('权重重要性')
        plt.grid(True, alpha=0.3)
        
        # 标注最大权重位置
        max_pos = np.argmax(time_importance)
        plt.axvline(x=max_pos, color='red', linestyle='--', 
                   label=f'最大滞后: {max_pos}')
        plt.legend()
        
        # 绘制累积权重分布
        plt.subplot(2, 1, 2)
        cumulative_weights = np.cumsum(time_importance) / np.sum(time_importance)
        plt.plot(range(self.kernel_size), cumulative_weights, 'o-', color='green')
        plt.title('累积权重分布')
        plt.xlabel('时间滞后位置')
        plt.ylabel('累积权重比例')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%分位线')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_causal_report(self, save_path=None, method='max_weight'):
        """生成因果关系分析报告"""
        lag_matrix, weight_importance = self.extract_causal_lags(method=method)
        causal_relationships = self.get_significant_causal_relationships(
            lag_matrix=lag_matrix, weight_importance=weight_importance
        )
        
        report = []
        report.append("=" * 80)
        report.append("因果滞后分析报告")
        report.append("=" * 80)
        report.append(f"分析方法: {method}")
        report.append(f"序列数量: {self.series_num}")
        report.append(f"卷积核大小: {self.kernel_size}")
        report.append(f"显著因果关系数量: {len(causal_relationships)}")
        report.append("")
        
        report.append("显著因果关系详情:")
        report.append("-" * 80)
        for i, rel in enumerate(causal_relationships, 1):
            report.append(f"{i:2d}. {rel['source']} → {rel['target']}")
            report.append(f"    滞后步数: {rel['lag']}")
            report.append(f"    权重重要性: {rel['weight_importance']:.4f}")
            report.append("")
        
        # 统计信息
        if causal_relationships:
            lags = [rel['lag'] for rel in causal_relationships]
            weights = [rel['weight_importance'] for rel in causal_relationships]
            
            report.append("统计摘要:")
            report.append("-" * 40)
            report.append(f"平均滞后步数: {np.mean(lags):.2f}")
            report.append(f"滞后步数标准差: {np.std(lags):.2f}")
            report.append(f"最常见滞后步数: {max(set(lags), key=lags.count)}")
            report.append(f"平均权重重要性: {np.mean(weights):.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text, lag_matrix, weight_importance, causal_relationships

# 使用示例
def analyze_causal_lags(model, series_names=None, save_dir=None):
    """
    完整的因果滞后分析流程
    
    Args:
        model: 训练好的MutiTS_GC模型
        series_names: 序列名称列表
        save_dir: 保存结果的目录
    """
    # 创建分析器
    analyzer = CausalLagAnalyzer(model, series_names)
    
    # 生成分析报告
    report_text, lag_matrix, weight_importance, causal_relationships = analyzer.generate_causal_report()
    print(report_text)
    
    # 绘制热力图
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        heatmap_path = os.path.join(save_dir, 'causal_lag_heatmap.png')
        report_path = os.path.join(save_dir, 'causal_lag_report.txt')
        
        analyzer.plot_lag_heatmap(lag_matrix, weight_importance, save_path=heatmap_path)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    else:
        analyzer.plot_lag_heatmap(lag_matrix, weight_importance)
    
    # 可选：绘制具体因果关系的权重分布
    if causal_relationships:
        top_relationship = causal_relationships[0]  # 最强的因果关系
        if save_dir:
            weight_dist_path = os.path.join(save_dir, 'weight_distribution.png')
            analyzer.plot_weight_distribution(
                top_relationship['target_idx'], 
                top_relationship['source_idx'],
                save_path=weight_dist_path
            )
        else:
            analyzer.plot_weight_distribution(
                top_relationship['target_idx'], 
                top_relationship['source_idx']
            )
    
    return analyzer, lag_matrix, weight_importance, causal_relationships

# 在你的主函数中添加分析代码
def add_causal_analysis_to_main():
    """在主函数中添加因果分析的示例代码"""
    
    # 在训练完成后添加以下代码：
    
    # 假设你有序列名称
    series_names = [f"变量{i+1}" for i in range(SERIES_NUM)]  # 或者使用实际的变量名
    
    # 进行因果滞后分析
    print("\n" + "="*50)
    print("开始因果滞后分析...")
    print("="*50)
    
    analyzer, lag_matrix, weight_importance, causal_relationships = analyze_causal_lags(
        model=model,
        series_names=series_names,
        save_dir=save_dir / "causal_analysis"
    )
    
    # 可以进一步分析特定的因果关系
    if causal_relationships:
        print(f"\n发现 {len(causal_relationships)} 个显著因果关系")
        print("前5个最强的因果关系:")
        for i, rel in enumerate(causal_relationships[:5], 1):
            print(f"{i}. {rel['source']} → {rel['target']}, 滞后: {rel['lag']}, 重要性: {rel['weight_importance']:.4f}")
    
    return analyzer