import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
from torch.cuda.amp import autocast, GradScaler

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, use_sparse_attention=False):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.use_sparse_attention = use_sparse_attention

    def forward(self, value, key, query, mask, memory=None):
        # Optional memory attention (for multi-turn dialogue)
        if memory is not None:
            query = torch.cat([memory, query], dim=0)  # Concatenate memory and current query for context

        # Optionally implement sparse attention for longer sequences
        if self.use_sparse_attention:
            attention = self.sparse_attention(query, key, value, mask)
        else:
            attention = self.attention(query, key, value, attn_mask=mask)[0]
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

    def sparse_attention(self, query, key, value, mask):
        # Placeholder for future sparse attention implementation
        return self.attention(query, key, value, attn_mask=mask)[0]


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, dropout, forward_expansion, layer_sharing=False):
        super(TransformerEncoder, self).__init__()
        self.layer_sharing = layer_sharing
        self.memory = None  # Memory to store conversation history for multi-turn dialogues

        if layer_sharing:
            self.shared_layer = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        else:
            self.layers = nn.ModuleList(
                [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
            )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask, memory=None):
        # Incorporate memory (previous conversation context) in the encoder
        if self.layer_sharing:
            for _ in range(len(self.layers)):
                x = self.shared_layer(x, x, x, mask, memory=memory)
        else:
            for layer in self.layers:
                x = layer(x, x, x, mask, memory=memory)

        # Store output as memory for future dialogue turns
        self.memory = x
        return self.norm(x), self.memory  # Return both output and memory for dialogue tracking


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, dropout, forward_expansion):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, enc_output, src_mask, tgt_mask, memory=None):
        for layer in self.layers:
            x = layer(x, enc_output, x, tgt_mask, memory=memory)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, dropout, forward_expansion, num_classes, use_mixed_precision=False):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(embed_size, num_layers, heads, dropout, forward_expansion)
        self.decoder = TransformerDecoder(embed_size, num_layers, heads, dropout, forward_expansion)
        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None

    def forward(self, src, tgt, src_mask, tgt_mask, memory=None):
        with autocast(enabled=self.use_mixed_precision):
            enc_output, memory = self.encoder(src, src_mask, memory)  # Pass memory for multi-turn context tracking
            dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask, memory=memory)
            output = self.fc_out(self.dropout(dec_output.mean(dim=1)))
        return output, memory  # Return both output and memory to continue dialogue tracking

    def prune_model(self, amount=0.2):
        # Apply global pruning
        parameters_to_prune = [(module, 'weight') for module in self.modules() if isinstance(module, nn.Linear)]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)


# Example usage with dialogue context:
if __name__ == "__main__":
    embed_size = 256
    num_layers = 6
    heads = 8
    dropout = 0.1
    forward_expansion = 4
    num_classes = 10
    seq_len = 20
    batch_size = 32
    use_mixed_precision = True

    src = torch.rand((seq_len, batch_size, embed_size))  # Source sequence
    tgt = torch.rand((seq_len, batch_size, embed_size))  # Target sequence
    src_mask = None
    tgt_mask = None
    memory = None  # Initialize memory

    model = Transformer(embed_size, num_layers, heads, dropout, forward_expansion, num_classes, use_mixed_precision)
    output, memory = model(src, tgt, src_mask, tgt_mask, memory)  # Output and updated memory

    print(output.shape)  # (batch_size, num_classes)

    # Example of pruning
    model.prune_model(amount=0.3)
