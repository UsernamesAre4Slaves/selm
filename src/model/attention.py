import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, sparse_attention=False, dropout=0.1, context_size=100):
        """
        Self-Attention mechanism with optional sparse attention and dialogue history integration.
        
        Args:
            embed_size (int): Dimension of the embedding vector.
            heads (int): Number of attention heads.
            sparse_attention (bool): Whether to use sparse attention.
            dropout (float): Dropout rate.
            context_size (int): Maximum size of the dialogue context to be incorporated.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.sparse_attention = sparse_attention
        self.context_size = context_size  # For dialogue context handling
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        # Define linear layers for key, query, and value projections
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        # Optional sparse attention mask
        if sparse_attention:
            self.sparse_mask = nn.Parameter(torch.ones(self.heads, embed_size, embed_size), requires_grad=False)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask=None, dialogue_history=None):
        """
        Forward pass for self-attention mechanism with optional dialogue history.

        Args:
            values (Tensor): Values tensor.
            keys (Tensor): Keys tensor.
            query (Tensor): Query tensor.
            mask (Tensor, optional): Mask tensor for attention.
            dialogue_history (Tensor, optional): Dialogue history tensor (for context-based attention).
        
        Returns:
            Tensor: Output of the attention mechanism.
        """
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Incorporate dialogue history into the attention mechanism if provided
        if dialogue_history is not None:
            dialogue_history = dialogue_history[:, -self.context_size:]  # Limit to context size
            query = torch.cat([dialogue_history, query], dim=1)

        # Reshape for multi-head attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query.shape[1], self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply sparse attention mask if enabled
        if self.sparse_attention:
            energy = energy * self.sparse_mask.unsqueeze(0)  # Apply sparse mask
        
        # Apply attention mask (e.g., padding mask)
        if mask is not None:
            energy.masked_fill_(mask == 0, float("-1e20"))

        # Attention scores and dropout
        attention = F.softmax(energy / (self.embed_size ** 0.5), dim=-1)
        attention = self.dropout(attention)

        # Aggregate values based on attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query.shape[1], self.heads * self.head_dim)

        # Final output projection
        out = self.fc_out(out)
        return out

class AttentionClassifier(nn.Module):
    def __init__(self, embed_size, heads, num_classes, sparse_attention=False, dropout=0.1, context_size=100):
        """
        Classification model using attention mechanism with dialogue history.
        
        Args:
            embed_size (int): Dimension of the embedding vector.
            heads (int): Number of attention heads.
            num_classes (int): Number of output classes.
            sparse_attention (bool): Whether to use sparse attention.
            dropout (float): Dropout rate.
            context_size (int): Maximum size of the dialogue context.
        """
        super(AttentionClassifier, self).__init__()
        self.attention = SelfAttention(embed_size, heads, sparse_attention, dropout, context_size)
        self.fc = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, dialogue_history=None):
        """
        Forward pass for classification with context handling.

        Args:
            x (Tensor): Input tensor (e.g., word embeddings).
            mask (Tensor, optional): Attention mask.
            dialogue_history (Tensor, optional): Tensor representing past dialogue context.
        
        Returns:
            Tensor: Logits for each class.
        """
        x = self.attention(x, x, x, mask, dialogue_history)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

def attention_efficiency_evaluation(model, data_loader, device):
    """
    Evaluate the efficiency of the attention model in terms of sparse attention utilization.
    
    Args:
        model (nn.Module): The attention model.
        data_loader (DataLoader): DataLoader with test data.
        device (torch.device): Device for inference (e.g., 'cuda', 'cpu').
    
    Returns:
        float: Efficiency score based on model's ability to sparsely attend to key tokens.
    """
    model.to(device)
    model.eval()
    
    sparse_attention_utilization = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None).to(device)

            # Forward pass
            _ = model(input_ids, attention_mask)

            # Evaluate sparse attention utilization if enabled
            if hasattr(model.attention, 'sparse_mask'):
                sparse_attention_utilization += model.attention.sparse_mask.sum().item()

            total_batches += 1
    
    return sparse_attention_utilization / total_batches

if __name__ == "__main__":
    # Example usage of attention model with context handling
    embed_size = 512
    heads = 8
    num_classes = 10
    sparse_attention = True  # Enable sparse attention
    context_size = 50  # Dialogue history context size

    model = AttentionClassifier(embed_size, heads, num_classes, sparse_attention, context_size=context_size)
    
    # Example input data (replace with actual data)
    input_data = torch.rand((32, 100, embed_size))  # Batch size of 32, sequence length of 100
    dialogue_history = torch.rand((32, 50, embed_size))  # Simulated dialogue history

    # Perform forward pass with context
    output = model(input_data, dialogue_history=dialogue_history)
    print("Output shape:", output.shape)  # Should be (32, num_classes)
