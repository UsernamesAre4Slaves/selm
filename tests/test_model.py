import unittest
import torch
import torch.nn as nn
from src.model.transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # Set up a simple Transformer model for testing
        self.embed_size = 256
        self.num_layers = 4
        self.heads = 8
        self.dropout = 0.1
        self.forward_expansion = 4
        self.num_classes = 10
        self.model = Transformer(
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout,
            forward_expansion=self.forward_expansion,
            num_classes=self.num_classes
        )

    def test_transformer_initialization(self):
        """Test if Transformer model initializes correctly."""
        self.assertIsInstance(self.model, Transformer)
        self.assertEqual(self.model.embed_size, self.embed_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.heads, self.heads)
        self.assertEqual(self.model.dropout, self.dropout)
        self.assertEqual(self.model.forward_expansion, self.forward_expansion)
        self.assertEqual(self.model.num_classes, self.num_classes)

    def test_transformer_forward(self):
        """Test the forward pass of the Transformer model."""
        self.model.train()
        src = torch.randn(32, 10, self.embed_size)  # (batch_size, sequence_length, embed_size)
        tgt = torch.randn(32, 10, self.embed_size)  # (batch_size, sequence_length, embed_size)

        output = self.model(src, tgt, src_mask=None, tgt_mask=None)
        self.assertEqual(output.size(), (32, 10, self.num_classes))

    def test_transformer_loss(self):
        """Test the loss computation for the Transformer model."""
        self.model.train()
        src = torch.randn(32, 10, self.embed_size)  # (batch_size, sequence_length, embed_size)
        tgt = torch.randn(32, 10, self.embed_size)  # (batch_size, sequence_length, embed_size)
        targets = torch.randint(0, self.num_classes, (32, 10))  # (batch_size, sequence_length)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        optimizer.zero_grad()
        output = self.model(src, tgt, src_mask=None, tgt_mask=None)
        output = output.view(-1, self.num_classes)  # Flatten for CrossEntropyLoss
        targets = targets.view(-1)  # Flatten targets
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        self.assertGreater(loss.item(), 0)

if __name__ == "__main__":
    unittest.main()
