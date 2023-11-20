from torch import nn
import torch
from torch.nn import functional as F

from wml_ai_model_managers.text_model_manager_zero.head import WMLHead


class WMLMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size,dropout,n_embd,block_size):
        super().__init__()
        self.heads = nn.ModuleList(
          [WMLHead(n_embd,head_size,block_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out


