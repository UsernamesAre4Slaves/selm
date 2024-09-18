import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import logging
from torch.quantization import QuantStub, DeQuantStub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_pruning(model, pruning_method='l1_unstructured', amount=0.2, structured=False, global_pruning=False):
    """
    Apply pruning to the model's layers, with optional structured and global pruning.
    
    Args:
        model (nn.Module): The model to prune.
        pruning_method (str): The pruning method ('l1_unstructured', 'random_unstructured', 'l1_structured').
        amount (float): The fraction of weights to prune.
        structured (bool): Whether to apply structured pruning (e.g., channels).
        global_pruning (bool): Whether to apply global pruning across the entire model.
    """
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if structured and pruning_method == 'l1_structured':
                prune.l1_structured(module, name='weight', amount=amount, dim=0)
                logger.info(f"Applied structured L1 pruning to {name}")
            else:
                parameters_to_prune.append((module, 'weight'))
    
    if global_pruning:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured if pruning_method == 'l1_unstructured' else prune.RandomUnstructured,
            amount=amount
        )
        logger.info(f"Applied global {pruning_method} pruning.")
    else:
        for module, name in parameters_to_prune:
            if pruning_method == 'l1_unstructured':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif pruning_method == 'random_unstructured':
                prune.random_unstructured(module, name='weight', amount=amount)
            logger.info(f"Applied {pruning_method} pruning to {module}")

def sensitivity_based_pruning(model, sensitivity_map, base_amount=0.1):
    """
    Prunes each layer based on a pre-defined sensitivity map.
    
    Args:
        model (nn.Module): The model to prune.
        sensitivity_map (dict): A map of layer names to pruning amounts.
        base_amount (float): Base amount of pruning to apply if no specific sensitivity is provided.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune_amount = sensitivity_map.get(name, base_amount)
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            logger.info(f"Applied sensitivity-based pruning to {name}, amount: {prune_amount}")

def remove_pruning(model):
    """
    Remove pruning masks and reapply the original weights.

    Args:
        model (nn.Module): The model with pruning applied.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.remove(module, 'weight')
            logger.info(f"Removed pruning from {name}")

def apply_quantization(model, dtype=torch.qint8, qconfig='fbgemm'):
    """
    Apply post-training quantization to the model.

    Args:
        model (nn.Module): The model to quantize.
        dtype (torch.dtype): The data type for quantization (e.g., torch.qint8).
        qconfig (str): The quantization configuration ('fbgemm' or 'qnnpack').
    """
    model.eval()
    
    # Apply quantization configuration
    if qconfig == 'fbgemm':
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    elif qconfig == 'qnnpack':
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    else:
        raise ValueError(f"Unsupported quantization configuration: {qconfig}")

    # Prepare the model for quantization
    model = torch.quantization.prepare(model, inplace=True)

    # Calibration step with dummy data
    dummy_input = torch.randn(1, 20, 256)  # Modify this to fit your model's input size
    model(dummy_input)

    # Convert to quantized model
    model = torch.quantization.convert(model, inplace=True)
    logger.info(f"Applied quantization with dtype {dtype} and qconfig {qconfig}")

def prune_and_quantize(model, pruning_method='l1_unstructured', amount=0.2, sensitivity_map=None):
    """
    Apply pruning and quantization sequentially.

    Args:
        model (nn.Module): The model to prune and quantize.
        pruning_method (str): The pruning method to use.
        amount (float): The fraction of weights to prune.
        sensitivity_map (dict): Sensitivity map for sensitivity-based pruning.
    """
    # Apply sensitivity-based pruning if provided
    if sensitivity_map:
        sensitivity_based_pruning(model, sensitivity_map)
    else:
        apply_pruning(model, pruning_method=pruning_method, amount=amount)
    
    # Apply quantization
    apply_quantization(model)

def main():
    # Define a transformer model (use any model suited for your task)
    model = Transformer(embed_size=256, num_layers=4, heads=8, dropout=0.1, forward_expansion=4, num_classes=10)
    model = model.to('cpu')  # For quantization purposes

    # Sensitivity map (example): Specifies pruning amounts per layer
    sensitivity_map = {
        "layer1": 0.1,
        "layer2": 0.3,
        "layer3": 0.2,
        "layer4": 0.15
    }

    # Apply pruning and quantization
    prune_and_quantize(model, pruning_method='l1_unstructured', amount=0.2, sensitivity_map=sensitivity_map)

    # Optionally, save the pruned and quantized model
    torch.save(model.state_dict(), 'pruned_quantized_transformer_model.pth')
    logger.info("Model pruning and quantization completed successfully.")

if __name__ == "__main__":
    main()
