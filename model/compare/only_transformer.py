import numpy as np
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(-2), :]

class Embedding(nn.Module):
    def __init__(self, series_num, input_window, feature_dim, d_model, drop_prob, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # Linear transformation to embed input features to d_model dimension
        self.time_embed = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x):
        # x: [batch, series, seq_len, features]
        batch, n_series, seq_len, feat_dim = x.shape
        
        # Reshape to combine batch and series dimensions for linear layer
        x_reshaped = x.reshape(-1, seq_len, feat_dim)  # [batch*series, seq_len, features]
        
        # Apply linear transformation
        x_embedded = self.time_embed(x_reshaped)  # [batch*series, seq_len, d_model]
        
        # Reshape back to include series dimension
        x_embedded = x_embedded.reshape(batch, n_series, seq_len, -1)  # [batch, series, seq_len, d_model]
        
        # Apply positional encoding
        x_embedded = self.pos_encoder(x_embedded)
        
        return self.dropout(x_embedded)

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 下三角因果掩码
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 5000, 5000)))

    def forward(self, x):
        # x: [batch, series, seq_len, d_model]
        batch, n_series, seq_len, _ = x.shape
        q = self.Wq(x).view(batch, n_series, seq_len, self.n_head, self.d_k).transpose(2, 3)
        k = self.Wk(x).view(batch, n_series, seq_len, self.n_head, self.d_k).transpose(2, 3)
        v = self.Wv(x).view(batch, n_series, seq_len, self.n_head, self.d_k).transpose(2, 3)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = self.mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(2, 3).contiguous().view(batch, n_series, seq_len, self.d_model)
        return self.out(context)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = CausalAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class PredictModel2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Extract configuration parameters
        self.input_window = config['input_window']
        self.output_window = config['output_window']
        self.series_num = config['series_num']  # 不同变量/时间序列的数量
        self.feature_dim = config['feature_dim']
        self.output_dim = config['output_dim']  # 每个预测变量的输出维度，默认为1
        self.d_model = config['d_model']
        
        # Initialize embedding layer
        self.embedding = Embedding(
            series_num=self.series_num,
            input_window=self.input_window,
            feature_dim=self.feature_dim, 
            d_model=self.d_model, 
            drop_prob=config['drop_prob'], 
            device=config['device']
        )
        
        # Initialize encoder layers
        self.encoder = nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model, 
                n_head=config['n_head'], 
                dim_feedforward=config['ffn_hidden'], 
                dropout=config['drop_prob']
            )
            for _ in range(config['n_layers'])
        ])
        
        # LSTM decoder for sequence generation
        self.decoder = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=2,
            dropout=config['drop_prob'],
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.output_dim)
        )

    def forward(self, x):
        # x: [batch, input_window, series_num, feature_dim]
        # Permute input to match our model's expected format: [batch, series, seq_len, features]
        x = x.permute(0, 2, 1, 3)
        
        # Pass through embedding layer
        emb = self.embedding(x)  # [batch, series, seq_len, d_model]
        
        # Pass through encoder layers
        for layer in self.encoder:
            emb = layer(emb)
        
        # Take last timestep for decoder input
        dec_in = emb[:, :, -1, :].unsqueeze(2)  # [batch, series, 1, d_model]
        
        outputs = []
        hidden = None  # LSTM hidden state (initialized as None)
        
        # Autoregressive generation of output sequence
        for _ in range(self.output_window):
            # Flatten batch and series dimensions for LSTM
            batch_size, n_series, seq_len, d_model = dec_in.shape
            reshaped_input = dec_in.reshape(batch_size * n_series, seq_len, d_model)
            
            # Process through LSTM decoder
            if hidden is None:
                out, hidden = self.decoder(reshaped_input)
            else:
                out, hidden = self.decoder(reshaped_input, hidden)
            
            # Reshape output and get predictions
            out = out.reshape(batch_size, n_series, seq_len, d_model)
            pred = self.fc(out)  # [batch, series, seq_len, output_dim]
            outputs.append(pred)
            
            # Prepare input for next timestep
            if self.output_dim < self.d_model:
                proj_pred = torch.zeros(batch_size, n_series, seq_len, self.d_model, device=x.device)
                proj_pred[:, :, :, :self.output_dim] = pred
            else:
                proj_pred = pred[:, :, :, :self.d_model]
                
            dec_in = proj_pred
        
        # Stack outputs and reshape to match expected output format [batch, output_window, series_num, output_dim]
        predictions = torch.cat(outputs, dim=2)  # [batch, series, output_window, output_dim]
        predictions = predictions.permute(0, 2, 1, 3)  # [batch, output_window, series, output_dim]
        
        return predictions