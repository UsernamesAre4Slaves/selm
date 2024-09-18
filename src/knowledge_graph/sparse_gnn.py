import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_sparse import SparseTensor
from src.knowledge_graph.graph_utils import load_knowledge_graph

class SparseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, conv_type='GCN', dropout=0.5):
        """
        Sparse Graph Neural Network (Sparse GNN) using GCN or SAGE convolutions.
        
        Args:
            input_dim (int): Dimensionality of input node features.
            hidden_dim (int): Hidden layer size.
            output_dim (int): Output dimensionality (e.g., number of classes).
            num_layers (int): Number of GNN layers.
            conv_type (str): Type of GNN convolution ('GCN' or 'SAGE').
            dropout (float): Dropout rate for training.
        """
        super(SparseGNN, self).__init__()
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize layers
        self.layers = nn.ModuleList()
        
        if conv_type == 'GCN':
            self.layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(GCNConv(hidden_dim, output_dim))
        elif conv_type == 'SAGE':
            self.layers.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.layers.append(SAGEConv(hidden_dim, output_dim))
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(self, x, edge_index):
        """
        Forward pass of SparseGNN.
        
        Args:
            x (Tensor): Input node features.
            edge_index (Tensor or SparseTensor): Sparse adjacency matrix or edge list.
        
        Returns:
            Tensor: Output node embeddings or class predictions.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(edge_index, SparseTensor):
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index)
            
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

def load_sparse_graph(file_path):
    """
    Load sparse graph and features for training or inference.
    
    Args:
        file_path (str): Path to the knowledge graph data.
        
    Returns:
        Tensor: Node features.
        SparseTensor: Sparse adjacency matrix.
    """
    edge_index, node_features = load_knowledge_graph(file_path)
    
    # Convert edge_index to SparseTensor
    num_nodes = node_features.size(0)
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    adj_matrix = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
    
    return node_features, adj_matrix

def train_sparse_gnn(model, data_loader, optimizer, device):
    """
    Train the SparseGNN model.
    
    Args:
        model (nn.Module): The SparseGNN model.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
        
    Returns:
        float: Training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        optimizer.zero_grad()
        node_features, adj_matrix, labels = batch
        node_features = node_features.to(device)
        adj_matrix = adj_matrix.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(node_features, adj_matrix)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate_sparse_gnn(model, data_loader, device):
    """
    Evaluate the SparseGNN model.
    
    Args:
        model (nn.Module): The SparseGNN model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run the evaluation on (e.g., 'cpu' or 'cuda').
        
    Returns:
        float: Evaluation accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            node_features, adj_matrix, labels = batch
            node_features = node_features.to(device)
            adj_matrix = adj_matrix.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(node_features, adj_matrix)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total

if __name__ == '__main__':
    # Example usage of SparseGNN

    # Configurations
    config = {
        'input_dim': 128,        # Example input dimension
        'hidden_dim': 64,        # Hidden dimension
        'output_dim': 10,        # Number of output classes
        'num_layers': 2,         # Number of GNN layers
        'conv_type': 'GCN',      # Type of convolution (GCN or SAGE)
        'dropout': 0.5           # Dropout rate
    }
    
    # Load graph data
    graph_file_path = 'data/knowledge_graph/graph_data.csv'
    node_features, adj_matrix = load_sparse_graph(graph_file_path)

    # Initialize model
    model = SparseGNN(config['input_dim'], config['hidden_dim'], config['output_dim'],
                      num_layers=config['num_layers'], conv_type=config['conv_type'],
                      dropout=config['dropout'])

    # Example training loop (replace with actual DataLoader and optimizer)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # train_sparse_gnn(model, data_loader, optimizer, device='cuda')

    # Example evaluation loop (replace with actual DataLoader)
    # accuracy = evaluate_sparse_gnn(model, data_loader, device='cuda')
    # print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
