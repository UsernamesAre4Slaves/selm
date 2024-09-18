import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_random_graph(num_nodes, num_edges, num_features, num_classes):
    """
    Creates a random graph for demonstration purposes.

    Parameters:
    - num_nodes (int): Number of nodes in the graph.
    - num_edges (int): Number of edges in the graph.
    - num_features (int): Number of node features.
    - num_classes (int): Number of target classes.

    Returns:
    - Data: A PyTorch Geometric Data object containing the random graph.
    """
    x = torch.randn((num_nodes, num_features))  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)  # Edge indices
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)  # Node labels
    batch = torch.zeros(num_nodes, dtype=torch.long)  # Batch vector

    return Data(x=x, edge_index=edge_index, y=y, batch=batch)

def preprocess_data(data, normalization=True):
    """
    Preprocesses graph data by normalizing node features.

    Parameters:
    - data (Data): A PyTorch Geometric Data object.
    - normalization (bool): Whether to normalize node features.

    Returns:
    - Data: A PyTorch Geometric Data object with preprocessed node features.
    """
    if normalization:
        # Normalize node features
        x = data.x.numpy()
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        data.x = torch.tensor(x, dtype=torch.float)

    return data

def split_data(data, train_ratio=0.8):
    """
    Splits data into training and testing sets.

    Parameters:
    - data (Data): A PyTorch Geometric Data object.
    - train_ratio (float): Ratio of data to use for training.

    Returns:
    - Data, Data: Training and testing Data objects.
    """
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(num_nodes * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Mask for edge_index
    edge_mask_train = np.isin(data.edge_index, train_indices).all(axis=0)
    edge_mask_test = np.isin(data.edge_index, test_indices).all(axis=0)

    train_data = Data(
        x=data.x[train_indices],
        edge_index=data.edge_index[:, edge_mask_train],
        y=data.y[train_indices],
        batch=data.batch[train_indices]
    )
    
    test_data = Data(
        x=data.x[test_indices],
        edge_index=data.edge_index[:, edge_mask_test],
        y=data.y[test_indices],
        batch=data.batch[test_indices]
    )

    return train_data, test_data

def get_dataloader(data, batch_size=32, shuffle=True):
    """
    Creates a DataLoader for a given Data object.

    Parameters:
    - data (Data): A PyTorch Geometric Data object.
    - batch_size (int): Batch size for the DataLoader.
    - shuffle (bool): Whether to shuffle the data.

    Returns:
    - DataLoader: A PyTorch DataLoader for the graph data.
    """
    from torch_geometric.data import DataLoader
    return DataLoader([data], batch_size=batch_size, shuffle=shuffle)
