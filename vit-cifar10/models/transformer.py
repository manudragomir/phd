import torch.nn as nn

from .attention import MultiHeadAttention


class Feedforward(nn.Module):
    def __init__(self, emb_size, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, num_heads, emb_size, dropout=0.2):
        super().__init__()

        assert emb_size % num_heads == 0

        head_size = emb_size // num_heads

        self.mah = MultiHeadAttention(
            head_size, num_heads, emb_size, dropout
        )

        self.ffwd = Feedforward(emb_size, dropout)

        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.mah(self.ln1(x))

        # Feedforward with residual connection
        x = x + self.ffwd(self.ln2(x))

        return x
