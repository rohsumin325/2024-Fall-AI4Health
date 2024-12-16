import os
import pandas as pd
import numpy as np
from torch.nn.functional import mse_loss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
import math
import random

seed = 42 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def get_attn_vanilla_decoder_mask(seq): ## vanilla mask: 원래 transformer decoding mask 방식
    subsequent_mask = torch.ones_like(seq).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) 
    return subsequent_mask

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, custom_weight=None):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0
        
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, input_dim)
        
        if custom_weight is not None:
            self.v_linear.weight.data = custom_weight.to(self.v_linear.weight.device)
            if self.v_linear.bias is None:
                self.v_linear.bias.data.zero_()
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, v)

        output = output.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        output = output.contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out(output)

        return output 

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, input_dim, dim_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.1, custom_weight=None):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, custom_weight=custom_weight)
        self.feed_forward = FeedForward(input_dim, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.1, custom_weight=None):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, custom_weight=custom_weight)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, custom_weight=custom_weight)
        self.feed_forward = FeedForward(input_dim, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, dec_mask):
        tgt2 = self.self_attn(tgt, tgt, tgt, dec_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

# Transformer Encoder-Decoder
class VAN_Transformer(nn.Module):
    def __init__(self, input_dim=68, num_heads=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, custom_weight=None):
        super(VAN_Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(input_dim, num_heads, dim_feedforward, dropout) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(input_dim, num_heads, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(input_dim, input_dim)
    
    def forward(self, src, tgt, dec_mask=None):
    
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
    
        for layer in self.encoder_layers:
            src = layer(src)
        
        memory = src
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, dec_mask)
        
        output = self.fc_out(tgt)
        return output

def upper_to_full(upper_vals, matrix_size=68):
    full_matrix = np.zeros((matrix_size, matrix_size))
    idx_upper = np.triu_indices(matrix_size, k=0)
    full_matrix[idx_upper] = upper_vals
    full_matrix += full_matrix.T
    np.fill_diagonal(full_matrix, full_matrix.diagonal() / 2)
    return full_matrix

def pearson_correlation_coefficient(x, y):
    batch_size = x.size(0)
    r_values = []

    for i in range(batch_size):
        xi = x[i].flatten()
        yi = y[i].flatten()

        mean_xi = xi.mean()
        mean_yi = yi.mean()
        xm = xi - mean_xi
        ym = yi - mean_yi
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
        r = r_num / r_den
        r = r if r_den > 1e-8 else 0.0
        
        r_values.append(r.item()) 
    return sum(r_values) / len(r_values)

def load_data(fold):
    fc_train = pd.read_csv(f"/HCP_folds/HCP_FC_final_train_fold_{fold}.csv")
    sc_train = pd.read_csv(f"/HCP_folds/HCP_SC_log10_train_fold_{fold}.csv")
    sc_train = sc_train.drop(columns=['Subjno'])
    fc_train = fc_train.drop(columns=['Subjno'])
    fc_test = pd.read_csv(f"/HCP_folds/HCP_FC_final_test_fold_{fold}.csv")
    sc_test = pd.read_csv(f"/HCP_folds/HCP_SC_log10_test_fold_{fold}.csv")
    sc_test = sc_test.drop(columns=['Subjno'])
    fc_test = fc_test.drop(columns=['Subjno'])
    return sc_train,sc_test, fc_train,fc_test

def normlize_data(all_data):
    eps = 1e-9
    all_targets = []
    all_inputs = []
    input_train = all_data[0]
    input_test = all_data[1]
    target_train = all_data[2]
    target_test = all_data[3]
    for index in range(len(input_train)):
        input_matrix = input_train.iloc[index]
        input_matrix = upper_to_full(input_matrix)
        all_inputs.append(input_matrix)
        
    for index in range(len(input_test)):
        input_matrix = input_test.iloc[index]
        input_matrix = upper_to_full(input_matrix)
        all_inputs.append(input_matrix)

    for index in range(len(target_train)):
        target_matrix = target_train.iloc[index]
        target_matrix = upper_to_full(target_matrix)
        all_targets.append(target_matrix)

    for index in range(len(target_test)):
        target_matrix = target_test.iloc[index]
        target_matrix = upper_to_full(target_matrix)
        all_targets.append(target_matrix)

    all_inputs = np.stack(all_inputs)
    all_targets = np.stack(all_targets)

    input_mean = all_inputs.mean((0, 1, 2), keepdims=True).squeeze(0).squeeze(0)
    input_std = all_inputs.std((0, 1, 2), keepdims=True).squeeze(0).squeeze(0)

    target_mean = all_targets.mean((0, 1, 2), keepdims=True).squeeze(0).squeeze(0)
    target_std = all_targets.std((0, 1, 2), keepdims=True).squeeze(0).squeeze(0)

    return (torch.from_numpy(input_mean) + eps, torch.from_numpy(input_std) + eps, torch.from_numpy(target_mean) + eps,
            torch.from_numpy(target_std) + eps)

class sm_Dataset(data.Dataset):
    def __init__(self, all_data, data_mean, train=True, test=False):
        self.train = train 
        self.test = test
        input_train = all_data[0]
        input_test = all_data[1]
        target_train = all_data[2]
        target_test = all_data[3]
        self.input_mean, self.input_std, self.feat_mean, self.feat_std = data_mean
        
        if self.train:
            self.input_data = input_train
            self.target_data = target_train
        elif self.test: #val
            self.input_data = input_test
            self.target_data = target_test
            
    def __getitem__(self, index):
        input_matrix = self.input_data.iloc[index]
        input_matrix = upper_to_full(input_matrix)
        
        target_matrix = self.target_data.iloc[index]
        target_matrix = upper_to_full(target_matrix)
    
        input_matrix = torch.from_numpy(input_matrix)
        input_matrix = (input_matrix - self.input_mean) / self.input_std

        target_matrix = torch.from_numpy(target_matrix)
        target_matrix = (target_matrix - self.feat_mean) / self.feat_std

        return input_matrix, target_matrix 

    def __len__(self):
        return len(self.input_data)

def get_loader(all_data, data_mean, training, test, batch_size=32, num_workers=4):
    dataset = sm_Dataset(all_data, data_mean, training, test)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_workers)
    return data_loader
