import torch
import torch.nn as nn
import math


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_head):
        super().__init__()
        self.embed_size = embed_size
        self.num_head = num_head
        self.w_q = nn.Linear(embed_size, num_head)
        self.w_k = nn.Linear(embed_size, num_head)
        self.w_v = nn.Linear(embed_size, num_head)
        self.w_combined = nn.Linear(embed_size, num_head)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, dim = q.shape
        head_dim = self.embed_size // self.num_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch_size, seq_len, self.num_head, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_head, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_head, head_dim).permute(0, 2, 1, 3)
        score = q @ k.transpose(2, 3) / math.sqrt(head_dim)
        if mask is not None:
            score = score.masked_fill(mask==0, -10000)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, dim)
        out = self.w_combined(score)
        return out


