import unittest
import torch
import torch.nn as nn
from src.knowledge_graph.gnn import GNN
from src.knowledge_graph.graph_utils import create_graph

class TestGNN(unittest.TestCase):
    def setUp(self):
        # Set up a simple GNN model for testing
        self.input_size = 10
        self.hidden_size = 16
        self.output_size = 2
        self.num_layers = 2
        self.model = GNN(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        self.graph_data = create_graph(num_nodes=5, num_edges=10)

    def test_gnn_initialization(self):
        """Test if GNN model initializes correctly."""
        self.assertIsInstance(self.model, GNN)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.output_size, self.output_size)
        self.assertEqual(self.model.num_layers, self.num_layers)

    def test_gnn_forward(self):
        """Test the forward pass of the GNN model."""
        self.model.train()
        node_features = torch.randn(5, self.input_size)
        edge_index = torch.randint(0, 5, (2, 10))  # Example edge indices

        output = self.model(node_features, edge_index)
        self.assertEqual(output.size(), (5, self.output_size))

    def test_gnn_loss(self):
        """Test the loss computation for the GNN model."""
        self.model.train()
        node_features = torch.randn(5, self.input_size)
        edge_index = torch.randint(0, 5, (2, 10))  # Example edge indices
        targets = torch.randint(0, self.output_size, (5,))  # Example targets

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        optimizer.zero_grad()
        output = self.model(node_features, edge_index)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        self.assertGreater(loss.item(), 0)

if __name__ == "__main__":
    unittest.main()
