import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj

class EnhancedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(EnhancedGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels * 3, out_channels)  # Concatenate mean, max, and add pooling
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # Apply dropout to edge_index for robust training
        edge_index, _ = dropout_adj(edge_index, p=self.dropout, training=self.training)
        
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global Pooling with mean, max, and add
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_add], dim=1)  # Concatenate mean, max, and add pooling
        
        # Final Fully Connected Layer
        x = self.fc(x)
        return x

def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total

def query_knowledge_graph(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        results = []
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            results.append(pred.cpu().numpy())
    return results

def main():
    # Example data
    num_nodes = 100
    num_edges = 200
    num_features = 16
    num_classes = 3

    # Create random data
    x = torch.randn((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Create a Data object
    data = Data(x=x, edge_index=edge_index, y=y, batch=batch)

    # Create DataLoader
    data_loader = DataLoader([data], batch_size=1)

    # Initialize model, optimizer, and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedGNN(in_channels=num_features, hidden_channels=32, out_channels=num_classes, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train and evaluate
    for epoch in range(10):
        loss = train(model, data_loader, optimizer, device)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    accuracy = evaluate(model, data_loader, device)
    print(f'Accuracy: {accuracy:.4f}')

    # Query the knowledge graph with the trained model
    results = query_knowledge_graph(model, data_loader, device)
    print('Knowledge Graph Query Results:', results)

if __name__ == "__main__":
    main()
