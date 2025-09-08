import torch
from torch import nn
import torch.nn.functional as F
import math

from torch import Tensor


# 将输入的词汇表索引转换为指定维度的Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size, padding_idx=1):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)


class PositionEmbedding(nn.Module):
    def __init__(self, embed_size, max_length, device):
        super().__init__()
        self.encoding = torch.zeros(max_length, embed_size, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_length, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, embed_size, step=2, device=device)
        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/embed_size)))
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/embed_size)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, dropout, device):
        super().__init__()
        self.token_embeddings = TokenEmbedding(vocab_size, embed_size)
        self.position_embeddings = PositionEmbedding(embed_size, max_length, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_embeddings = self.token_embeddings(x)
        position_embeddings = self.position_embeddings(x)
        output = self.dropout(token_embeddings + position_embeddings)
        return output