import torch
import logging
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixedPrecisionTraining:
    def __init__(self, initial_scale=2.**15, scale_factor=2.0, scale_back_factor=0.5, max_norm=1.0):
        """
        Initializes the mixed-precision training setup with PyTorch's AMP (Automatic Mixed Precision).
        
        Args:
            initial_scale (float): Initial scale factor for gradient scaling.
            scale_factor (float): Factor to increase the scale when no overflows are detected.
            scale_back_factor (float): Factor to decrease the scale when overflow is detected.
            max_norm (float): Maximum norm for gradient clipping.
        """
        self.scaler = GradScaler(init_scale=initial_scale)
        self.scale_factor = scale_factor
        self.scale_back_factor = scale_back_factor
        self.max_norm = max_norm  # For gradient clipping

    def apply_autocast(self, model, optimizer, data_loader, device, precision="mixed", log_interval=100):
        """
        Applies mixed-precision training to a model using autocast for forward passes
        and GradScaler for gradient scaling during backpropagation.

        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            data_loader (DataLoader): DataLoader providing the training data.
            device (str): Device to perform training on (CPU or GPU).
            precision (str): Precision mode, either "mixed" or "full".
            log_interval (int): Interval at which to log training progress.
        """
        model.train()
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Choose precision mode (mixed or full)
            if precision == "mixed":
                with autocast():
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
            else:
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)

            # Scale the loss and perform backward pass
            self.scaler.scale(loss).backward()

            # Unscale gradients and clip if necessary
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)

            # Step the optimizer
            self.scaler.step(optimizer)
            self.scaler.update()

            # Logging
            if batch_idx % log_interval == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item()}, Scale: {self.scaler.get_scale()}")

    def save_model(self, model, filepath):
        """
        Saves the model state dictionary.

        Args:
            model (nn.Module): The model to save.
            filepath (str): File path where the model state will be saved.
        """
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model state saved to {filepath}")

    def load_model(self, model, filepath):
        """
        Loads the model state dictionary from a file.

        Args:
            model (nn.Module): The model to load state into.
            filepath (str): File path from which the model state will be loaded.
        """
        model.load_state_dict(torch.load(filepath))
        logger.info(f"Model state loaded from {filepath}")

    def adjust_scale(self, overflow_detected):
        """
        Adjust the scale factor based on overflow detection.

        Args:
            overflow_detected (bool): Whether overflow was detected during training.
        """
        if overflow_detected:
            self.scaler.update_scale(self.scaler.get_scale() * self.scale_back_factor)
            logger.warning(f"Overflow detected, scale factor reduced to {self.scaler.get_scale()}")
        else:
            self.scaler.update_scale(self.scaler.get_scale() * self.scale_factor)
            logger.info(f"No overflow detected, scale factor increased to {self.scaler.get_scale()}")

    def validate(self, model, data_loader, device, precision="mixed"):
        """
        Evaluates the model on a validation dataset using mixed precision inference.
        
        Args:
            model (nn.Module): The model to evaluate.
            data_loader (DataLoader): DataLoader with validation data.
            device (str): Device to perform inference on (CPU or GPU).
            precision (str): Precision mode, either "mixed" or "full".
        
        Returns:
            float: Validation loss.
        """
        model.eval()
        validation_loss = 0.0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)

                if precision == "mixed":
                    with autocast():
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                else:
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)

                validation_loss += loss.item() * data.size(0)
                total += data.size(0)
        
        return validation_loss / total


# Example usage
if __name__ == "__main__":
    from torchvision.models import resnet18
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    # Example data and model setup
    model = resnet18(pretrained=False).to('cuda')
    optimizer = Adam(model.parameters(), lr=1e-3)
    data = torch.randn(100, 3, 224, 224)  # Example data
    target = torch.randint(0, 1000, (100,))  # Example target
    data_loader = DataLoader(TensorDataset(data, target), batch_size=16)

    # Initialize mixed precision training
    mpt = MixedPrecisionTraining()
    
    # Apply mixed precision training
    mpt.apply_autocast(model, optimizer, data_loader, device='cuda', precision='mixed')

    # Save and load model example
    mpt.save_model(model, 'model.pth')
    mpt.load_model(model, 'model.pth')
