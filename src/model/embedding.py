import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, padding_idx=0):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, embed_size)
        self.register_buffer('position', torch.arange(max_len).unsqueeze(1).float())
        self.embed_size = embed_size

    def forward(self, x):
        seq_len = x.size(1)
        positions = self.position[:seq_len, :]
        return x + self.embedding(positions)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=512, padding_idx=0):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size, padding_idx)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        return x
