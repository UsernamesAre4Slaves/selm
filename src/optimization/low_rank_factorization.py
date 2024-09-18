import numpy as np
import logging
from scipy.linalg import svd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tensorly.decomposition import parafac, tucker
import tensorly as tl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LowRankFactorization:
    def __init__(self, rank=None, factorization_method='svd'):
        """
        Initializes the low-rank factorization with a specified rank or method.

        Args:
            rank (int, optional): Desired rank for the factorization. If None, rank will be adapted automatically.
            factorization_method (str): Factorization method ('svd', 'tucker', 'cp').
        """
        self.rank = rank
        self.factorization_method = factorization_method

    def factorize(self, matrix):
        """
        Applies low-rank factorization to a given matrix using the specified method.

        Args:
            matrix (np.ndarray or torch.Tensor): The input matrix to factorize.

        Returns:
            Factorized components depending on the method (U, S, Vt for SVD).
        """
        logger.info(f"Performing low-rank factorization using {self.factorization_method.upper()}.")
        
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().detach().numpy()

        if self.factorization_method == 'svd':
            return self._svd_factorization(matrix)
        elif self.factorization_method == 'tucker':
            return self._tucker_factorization(matrix)
        elif self.factorization_method == 'cp':
            return self._cp_factorization(matrix)
        else:
            raise ValueError(f"Unknown factorization method: {self.factorization_method}")

    def _svd_factorization(self, matrix):
        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = svd(matrix, full_matrices=False)
        
        # Adapt rank based on singular values if not provided
        rank = self.rank if self.rank else self._adapt_rank(S)

        # Keep only the top `rank` singular values
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        return U, S, Vt

    def _tucker_factorization(self, matrix):
        # Perform Tucker decomposition
        rank = self.rank or [min(matrix.shape)]  # Adjust rank for Tucker
        core, factors = tucker(matrix, rank=rank)
        return core, factors

    def _cp_factorization(self, matrix):
        # Perform CP (CANDECOMP/PARAFAC) decomposition
        rank = self.rank or min(matrix.shape)
        weights, factors = parafac(matrix, rank=rank)
        return weights, factors

    def _adapt_rank(self, singular_values, threshold=0.95):
        """
        Automatically adapts the rank based on singular value distribution, retaining enough singular values
        to capture the desired percentage of energy.

        Args:
            singular_values (np.ndarray): Array of singular values.
            threshold (float): Percentage of energy to retain (default 95%).

        Returns:
            int: Adapted rank.
        """
        cumulative_energy = np.cumsum(singular_values) / np.sum(singular_values)
        rank = np.searchsorted(cumulative_energy, threshold)
        logger.info(f"Adapted rank based on singular values: {rank}")
        return rank

    def approximate_matrix(self, matrix):
        """
        Approximates a matrix using low-rank factorization.

        Args:
            matrix (np.ndarray or torch.Tensor): The matrix to approximate.

        Returns:
            Approximated matrix (np.ndarray or torch.Tensor).
        """
        if self.factorization_method == 'svd':
            U, S, Vt = self.factorize(matrix)
            approximated_matrix = np.dot(U, np.dot(np.diag(S), Vt))
        else:
            # For CP and Tucker, we reconstruct the matrix from factors
            core, factors = self.factorize(matrix)
            approximated_matrix = tl.kruskal_to_tensor((core, factors)) if self.factorization_method == 'cp' \
                else tl.tucker_to_tensor(core, factors)
        
        return approximated_matrix

    def apply_factorization(self, weights):
        """
        Applies low-rank factorization to model weights.

        Args:
            weights (torch.Tensor): The weight matrix to be factorized.

        Returns:
            The approximated weight matrix (torch.Tensor).
        """
        logger.info("Applying low-rank factorization to model weights.")
        approximated_weights = self.approximate_matrix(weights)
        return torch.tensor(approximated_weights, dtype=torch.float32)

    def integrate_into_model(self, model, layer_name):
        """
        Integrates low-rank factorized weights into a specified model layer.

        Args:
            model (torch.nn.Module): The model to update.
            layer_name (str): The name of the layer to replace with factorized weights.
        """
        layer = dict(model.named_parameters())[layer_name]
        original_weights = layer.data

        # Apply low-rank factorization
        approximated_weights = self.apply_factorization(original_weights)

        # Update the model layer weights
        layer.data = approximated_weights
        logger.info(f"Integrated low-rank factorized weights into {layer_name}")

    def factorize_all_layers(self, model, target_layer_types=(nn.Linear, nn.Conv2d)):
        """
        Applies low-rank factorization to all specified layers in a model.

        Args:
            model (torch.nn.Module): The model to apply factorization on.
            target_layer_types (tuple): Layer types to factorize (default: Linear and Conv2d).
        """
        for name, module in model.named_modules():
            if isinstance(module, target_layer_types):
                logger.info(f"Applying low-rank factorization to layer: {name}")
                factorized_weights = self.apply_factorization(module.weight.data)
                module.weight = nn.Parameter(factorized_weights)

    def visualize_singular_values(self, matrix):
        """
        Visualizes the singular values of the matrix for rank selection insights.

        Args:
            matrix (np.ndarray or torch.Tensor): The matrix to visualize.
        """
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().detach().numpy()

        U, S, Vt = self.factorize(matrix)
        
        # Plot Singular Values
        plt.figure(figsize=(6, 4))
        plt.plot(S, 'o-', label="Singular Values")
        plt.title('Singular Values Distribution')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example: Low-rank factorization on a random matrix
    weight_matrix = np.random.rand(100, 100)

    # Initialize low-rank factorization with SVD and adaptive rank
    lrf = LowRankFactorization(factorization_method='svd')
    
    # Apply factorization
    approximated_weights = lrf.apply_factorization(torch.tensor(weight_matrix, dtype=torch.float32))

    logger.info(f"Original matrix shape: {weight_matrix.shape}")
    logger.info(f"Approximated matrix shape: {approximated_weights.shape}")

    # Visualize the singular values
    lrf.visualize_singular_values(weight_matrix)
