import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, head_size, emb_size):
        super().__init__()
        
        self.head_size = head_size
        self.emb_size = emb_size
        
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)

    def forward(self, x):
        B, P, E = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        qk = q @ k.transpose(-1, -2) * (self.head_size ** -0.5)
        wei = F.softmax(qk, dim=-1)

        return wei @ v # B x P x H (where H = head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads, emb_size, dropout=0.2):
        super().__init__()

        assert num_heads * head_size == emb_size
        
        self.heads = nn.ModuleList([AttentionHead(head_size, emb_size) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))
