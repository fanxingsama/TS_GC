import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.utils.data as data

# ================= 1. 实验配置与文件夹管理 (完全保留) =================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = 'plot_comparison' 
OUTPUT_DIR = os.path.join(base_dir, timestamp)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f">>> ✅ 本次对比实验文件夹已创建: {OUTPUT_DIR}")

DATA_FILE = 'ots_series_data.csv'      
GRAPH_FILE = 'ots_GC_matrix.csv'       

# 超参数
WINDOW_SIZE = 30                   
EPOCHS = 200                       
LR = 0.001                         
HIDDEN_DIM = 64                    
PATIENCE = 10                       
PREDICT_LEN = 1500                  
BATCH_SIZE = 256
SEED = 42

# ### [修复建议 1] 建议先将 GAP 设为 1 测试模型能力，通了再改回 5
GAP = 1 

# === 日志部分 (完全保留) ===
LOG_FILE = os.path.join(OUTPUT_DIR, "experiment_log.txt")
log_content = f"""
=======================================================
Experiment Log: {timestamp}
=======================================================
[System Info]
Data File   : {DATA_FILE}
Graph File  : {GRAPH_FILE}

[Hyperparameters]
Window Size : {WINDOW_SIZE}
Predict Gap : {GAP}
Predict Len : {PREDICT_LEN}
Batch Size  : {BATCH_SIZE}
Learning Rate: {LR}
Epochs      : {EPOCHS}
Hidden Dim  : {HIDDEN_DIM}
Patience    : {PATIENCE}
Random Seed : {SEED}
=======================================================
"""
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(log_content)
print(f">>> 📄 实验参数已保存至: {LOG_FILE}")

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> 使用计算设备: {device}")

# ================= 2. 数据加载与预处理 (植入修复) =================
print(f">>> 正在读取数据: {DATA_FILE} ...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"找不到数据文件 {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.astype(str).str.strip()
sensor_names = list(df.columns)
num_nodes = len(sensor_names)
name2idx = {name: i for i, name in enumerate(sensor_names)}

# X 的归一化
scaler_x = StandardScaler()
data_scaled = scaler_x.fit_transform(df.values)

# ### [核心修复 2] 定义一个专门针对 Y (变化量) 的归一化器
# 原因：工业数据的变化量(Delta)通常极小(如0.0001)，模型会忽略它。
# 必须把它拉伸到标准正态分布，模型才能学到东西。
scaler_y = StandardScaler()

def create_dataset_numpy(data, window, gap=1):
    X, Y = [], []
    for i in range(len(data) - window - gap):
        X.append(data[i:i+window])
        # 预测的不是下一个点的值，而是未来gap步的值
        target_val = data[i + window + gap - 1]  # 这是正确的
        last_input_val = data[i + window - 1]
        Y.append(target_val - last_input_val)
    return np.array(X), np.array(Y)

# 1. 先生成 Numpy 格式数据
X_np, Y_np_raw = create_dataset_numpy(data_scaled, WINDOW_SIZE, gap=GAP)

# 2. 对 Y 进行独立归一化 (关键！)
# Y_np_raw 形状是 (Samples, Nodes)，scaler_y 需要这种形状
Y_np_scaled = scaler_y.fit_transform(Y_np_raw)

# 3. 转为 Tensor
X_all = torch.FloatTensor(X_np)
Y_all = torch.FloatTensor(Y_np_scaled)

train_size = int(len(X_all) * 0.8)
train_X = X_all[:train_size]
train_Y = Y_all[:train_size]
test_X = X_all[train_size:]
test_Y = Y_all[train_size:]

print(f">>> 数据集形状: Train={train_X.shape}, Test={test_X.shape}")

# ================= 3. 构建两个邻接矩阵 (保留你原来的加载逻辑) =================
def normalize_adj(adj):
    row_sum = np.sum(adj, axis=1)
    row_sum[row_sum == 0] = 1 
    return adj / row_sum[:, np.newaxis]

adj_base_np = np.eye(num_nodes, dtype=np.float32)
adj_base_norm = normalize_adj(adj_base_np)
adj_base_tensor = torch.from_numpy(adj_base_norm).float().to(device)

adj_causal_np = np.eye(num_nodes, dtype=np.float32) 

if os.path.exists(GRAPH_FILE):
    print(f">>> 正在加载因果图: {GRAPH_FILE} ...")
    graph_df = pd.read_csv(GRAPH_FILE, header=None, names=['src', 'dst', 'w'])
    count_edges = 0
    for _, row in graph_df.iterrows():
        try:
            # 保留你原本的处理逻辑
            src_val, dst_val = row['src'], row['dst']
            if isinstance(src_val, float) and src_val.is_integer(): src_val = int(src_val)
            if isinstance(dst_val, float) and dst_val.is_integer(): dst_val = int(dst_val)
            
            cause = str(src_val).strip()
            effect = str(dst_val).strip()
            weight = float(row['w'])
            
            if cause in name2idx and effect in name2idx:
                u, v = name2idx[cause], name2idx[effect]
                adj_causal_np[v, u] = weight 
                count_edges += 1
        except Exception as e:
            pass
    print(f">>> ✅ 因果图加载完成，有效边数: {count_edges}")
else:
    print(f">>> ⚠️ 警告: 未找到 {GRAPH_FILE}，对比实验将失效！")

adj_causal_norm = normalize_adj(adj_causal_np)
adj_causal_tensor = torch.from_numpy(adj_causal_norm).float().to(device)

# ================= 4. 模型定义 (保持不变) =================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

class SimpleTGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, adj_matrix):
        super(SimpleTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.register_buffer('adj', adj_matrix)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gcn = GraphConvolution(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.permute(0, 2, 1).contiguous().view(-1, x.shape[1], 1)
        _, h_n = self.gru(x_reshaped)
        node_features = h_n.squeeze(0).view(batch_size, self.num_nodes, -1)
        gcn_out = torch.relu(self.gcn(node_features, self.adj))
        prediction = self.out_proj(gcn_out).squeeze(-1)
        return prediction

# ================= 5. 训练函数 (保留逻辑) =================
def train_experiment(model_name, adj_matrix, train_x, train_y, test_x, test_y):
    print(f"\n>>> [启动训练] 模型: {model_name} (Mode: Mini-batch) ...")
    
    train_dataset = data.TensorDataset(train_x, train_y)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleTGCN(num_nodes, 1, HIDDEN_DIM, adj_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    early_stop_counter = 0
    best_state = None
    
    # 验证集处理：如果不大，直接放GPU；如果大，需要分批
    # 这里为了简便，假设验证集能放下；如果爆显存请改回 DataLoader
    test_x_gpu = test_x.to(device)
    test_y_gpu = test_y.to(device)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_out = model(test_x_gpu)
            val_loss = criterion(val_out, test_y_gpu).item()
        
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            best_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= PATIENCE:
            print(f"    - 早停于 Epoch {epoch+1}, Best Val Loss: {best_loss:.5f}")
            break
            
        if (epoch+1) % 10 == 0:
            print(f"    - Epoch {epoch+1}: Train Loss = {avg_train_loss:.5f}, Val Loss = {val_loss:.5f}")
            
    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss

# ================= 6. 执行对比训练 =================
model_base, loss_base = train_experiment(
    "Baseline (No Graph)", adj_base_tensor, train_X, train_Y, test_X, test_Y
)

model_causal, loss_causal = train_experiment(
    "Proposed (Causal Graph)", adj_causal_tensor, train_X, train_Y, test_X, test_Y
)

# ================= 7. 预测、评估与绘图 (植入修复) =================
print(">>> 正在生成对比结果...")

def predict_batch(model, input_data, batch_size=256):
    if not torch.is_tensor(input_data):
        input_data = torch.FloatTensor(input_data)
    dataset = torch.utils.data.TensorDataset(input_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_pred = model(batch_x)
            all_preds.append(batch_pred.cpu())
    return torch.cat(all_preds).numpy()

# ### 【核心修复：动态调整预测范围】 ###
# 总共可用的窗口数量 (约 2000 - WINDOW_SIZE)
max_samples = len(X_all)

# 如果 PREDICT_LEN 设置得太大或为 -1，则自动调整为最大可用长度
if PREDICT_LEN <= 0 or PREDICT_LEN > max_samples:
    actual_len = max_samples
    print(f"!!! 警告: PREDICT_LEN 超限，已自动调整为最大值: {actual_len}")
else:
    actual_len = PREDICT_LEN

print(f">>> 准备预测最近的 {actual_len} 个数据点 (总数据量: {len(df)})")

# 从全量窗口 X_all 中截取最后 actual_len 个作为模型输入
# 这样即使 PREDICT_LEN > 测试集长度，它也会自动去训练集里取窗口
eval_X = X_all[-actual_len:]

# 1. 模型预测 (得到的是 Scaled Delta)
pred_base_delta_scaled = predict_batch(model_base, eval_X)
pred_causal_delta_scaled = predict_batch(model_causal, eval_X)

# 2. 正确的反归一化流程
pred_base_delta_real = scaler_y.inverse_transform(pred_base_delta_scaled)
pred_causal_delta_real = scaler_y.inverse_transform(pred_causal_delta_scaled)

# 获取对应的上一个时刻的归一化值 (Last Observation)
last_obs_scaled = eval_X.cpu().numpy()[:, -1, :]

# 叠加差值并转回物理量纲
pred_base = scaler_x.inverse_transform(last_obs_scaled + pred_base_delta_real)
pred_causal = scaler_x.inverse_transform(last_obs_scaled + pred_causal_delta_real)

# ### 【关键点：真实值对齐】 ###
# 既然预测了 actual_len 个点，真实值也必须从原始数据末尾切出相同的长度
target_raw = df.values[-actual_len:]

# 验证长度是否一致，防止报错
assert len(target_raw) == len(pred_base), f"长度不匹配: True({len(target_raw)}) vs Pred({len(pred_base)})"

# 存储结果指标与绘图 (保持原有逻辑)
metrics_data = []

for i in range(num_nodes):
    sensor_name = sensor_names[i]
    
    y_true = target_raw[:, i]
    y_base = pred_base[:, i]
    y_causal = pred_causal[:, i]
    
    rmse_base = np.sqrt(mean_squared_error(y_true, y_base))
    rmse_causal = np.sqrt(mean_squared_error(y_true, y_causal))
    improve = (rmse_base - rmse_causal) / rmse_base * 100
    
    metrics_data.append([sensor_name, rmse_base, rmse_causal, improve])
    
    # === 绘图 ===
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Ground Truth', color='black', alpha=0.6, linewidth=1.5)
    plt.plot(y_base, label=f'Baseline (No Graph) RMSE: {rmse_base:.3f}', 
             color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
    plt.plot(y_causal, label=f'Proposed (Causal) RMSE: {rmse_causal:.3f}', 
             color='#d62728', linestyle='-', alpha=0.9, linewidth=2)
    
    color_text = 'green' if improve > 0 else 'red'
    title_str = (f"Sensor: {sensor_name}\n"
                 f"Improvement with Causal Graph: {improve:.2f}%")
    
    plt.title(title_str, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    safe_name = "".join([c for c in sensor_name if c.isalnum() or c=='_'])
    save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()

# 保存 CSV 表格
metrics_df = pd.DataFrame(metrics_data, columns=['Sensor', 'RMSE_Baseline', 'RMSE_Causal', 'Improvement_%'])
csv_path = os.path.join(OUTPUT_DIR, "comparison_metrics.csv")
metrics_df.to_csv(csv_path, index=False)

# === 追加日志 ===
avg_base_rmse = metrics_df['RMSE_Baseline'].mean()
avg_causal_rmse = metrics_df['RMSE_Causal'].mean()
avg_improve = metrics_df['Improvement_%'].mean()

result_log = f"""
[Final Results]
Average RMSE (Baseline) : {avg_base_rmse:.4f}
Average RMSE (Causal)   : {avg_causal_rmse:.4f}
Average Improvement     : {avg_improve:.2f}%

[Top 3 Improved Sensors]
{metrics_df.nlargest(3, 'Improvement_%')[['Sensor', 'Improvement_%']].to_string(index=False)}

[Top 3 Worsened Sensors]
{metrics_df.nsmallest(3, 'Improvement_%')[['Sensor', 'Improvement_%']].to_string(index=False)}
=======================================================
"""

with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(result_log)

print(f">>> 📄 最终实验结果已追加至日志: {LOG_FILE}")
print(f"\n>>> ✅ 对比分析完成!")
print(f">>> 结果已保存至文件夹: {OUTPUT_DIR}")
print(f">>> 指标汇总表: {csv_path}")
print(metrics_df.head())