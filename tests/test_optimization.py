import unittest
import torch
from src.optimization.optuna_tuner import OptunaTuner
from src.optimization.pruning import prune_model
from src.model.transformer import Transformer
import optuna

class TestOptimization(unittest.TestCase):
    def setUp(self):
        # Set up a simple Transformer model for optimization testing
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
        self.dummy_input = torch.randn(32, 10, self.embed_size)
        self.dummy_target = torch.randint(0, self.num_classes, (32, 10))

    def test_optuna_tuner(self):
        """Test Optuna Tuner for hyperparameter optimization."""
        def objective(trial):
            # Define hyperparameters to optimize
            embed_size = trial.suggest_int('embed_size', 128, 512)
            num_layers = trial.suggest_int('num_layers', 2, 6)
            heads = trial.suggest_int('heads', 4, 16)
            dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
            forward_expansion = trial.suggest_int('forward_expansion', 2, 8)
            num_classes = self.num_classes

            # Create model with the trial hyperparameters
            model = Transformer(
                embed_size=embed_size,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                num_classes=num_classes
            )
            # Dummy forward pass for testing
            output = model(self.dummy_input, self.dummy_input, None, None)
            loss = output.sum()  # Dummy loss calculation for the test

            return loss.item()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)

        self.assertGreater(len(study.best_params), 0)
        self.assertTrue(study.best_value < float('inf'))

    def test_prune_model(self):
        """Test model pruning."""
        pruned_model = prune_model(self.model, pruning_percentage=0.2)

        # Check if the model has fewer parameters after pruning
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        self.assertLess(pruned_params, original_params)
        self.assertGreater(original_params, pruned_params)
        
if __name__ == "__main__":
    unittest.main()
