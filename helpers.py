import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
from typing import Optional, List
from labml import tracker
from constants import * 
from collections import OrderedDict

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads, batch_size):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads
        self.batch_size = batch_size

        # Define linear layers for query, key, and value projections
        self.query_linear = nn.Linear(self.input_dim, self.input_dim)
        self.key_linear = nn.Linear(self.input_dim, self.input_dim)
        self.value_linear = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, array1, array2):
        # Project input arrays into query, key, and value spaces
        query = self.query_linear(array1)
        key = self.key_linear(array2)
        value = self.value_linear(array2)
        
        # Reshape tensors for multi-head attention
        query = query.view(self.batch_size, self.num_heads, self.head_dim)
        key = key.view(self.batch_size, self.num_heads, self.head_dim)
        value = value.view(self.batch_size, self.num_heads, self.head_dim)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Weighted sum of values using attention weights
        weighted_sum = torch.matmul(attention_weights, value)

        # Reshape and concatenate multi-head results
        weighted_sum = weighted_sum.view(self.batch_size, self.input_dim)
        
        return weighted_sum

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
            
            self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class CrossTransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, batch_size):
        super(CrossTransformerBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.batch_size = batch_size
        
        self.norm1 = nn.LayerNorm(self.hidden_d)

        self.mhsa = MultiHeadCrossAttention(num_heads=self.n_heads, input_dim=self.hidden_d, batch_size=self.batch_size)
        self.ff = FeedForward(d_model=self.hidden_d, d_ff=self.hidden_d)
        
        self.norm2 = nn.LayerNorm(hidden_d)

    def forward(self, x, y):
        out = self.norm1(x + self.mhsa(x, y))
        out = self.norm2(out + self.ff(out))
        
        return out

class SelfTransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, batch_size):
        super(SelfTransformerBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.batch_size = batch_size

        self.norm1 = nn.LayerNorm(self.hidden_d)

        self.mhsa = MultiHeadCrossAttention(num_heads=self.n_heads, input_dim=self.hidden_d, batch_size=self.batch_size)
        self.ff = FeedForward(d_model=self.hidden_d, d_ff=self.hidden_d)
        
        self.norm2 = nn.LayerNorm(hidden_d)

    def forward(self, x):
        out = self.norm1(x + self.mhsa(x, x))
        out = self.norm2(out + self.ff(out))
        
        return out