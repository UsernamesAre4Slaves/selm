import os
import torch
import torch.nn as nn
import torch.quantization
import torch_pruning as pruning
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model function
def load_model(model_path, num_labels):
    logging.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    return model

# Prune the model
def prune_model(model, pruning_ratio=0.2):
    logging.info(f"Pruning the model with pruning ratio: {pruning_ratio}")

    # Define the pruning strategy - pruning 20% of weights in each linear layer
    parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    # Perform magnitude-based pruning on linear layers
    pruning_strategy = pruning.MagnitudePruner(parameters_to_prune, pruning_ratio=pruning_ratio)
    pruned_model = pruning_strategy.prune(model)

    logging.info("Model pruning complete.")
    return pruned_model

# Quantize the model
def quantize_model(model):
    logging.info("Quantizing the model with dynamic quantization.")
    
    model.eval()  # Ensure the model is in evaluation mode before quantization

    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8  # Use 8-bit integer quantization
    )

    logging.info("Model quantization complete.")
    return quantized_model

# Validate the pruned and quantized model
def validate_model(model, dataset_name='imdb', batch_size=16):
    logging.info("Validating the pruned and quantized model.")
    
    # Load dataset for validation
    dataset = load_dataset(dataset_name, split='test')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    test_dataset = tokenized_dataset.remove_columns(['text'])
    
    # DataLoader for validation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to appropriate device (CPU for quantized models)
    device = torch.device('cpu')
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    # Validation loop
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    logging.info(f"Validation Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
    return accuracy, avg_loss

# Save the model
def save_model(model, save_path):
    # Ensure the model directory exists
    os.makedirs(save_path, exist_ok=True)
    # Save the model
    model.save_pretrained(save_path)
    logging.info(f"Model saved to {save_path}")

# Main function
def main():
    # Paths and configuration
    model_path = 'model_output/latest_model'  # Path to the latest model checkpoint
    num_labels = 2  # Number of labels for classification task
    output_path = 'model_output/pruned_quantized_model'  # Path to save pruned and quantized model
    pruning_ratio = 0.2  # Fraction of weights to prune
    dataset_name = 'imdb'  # Dataset for validation (can be replaced as needed)

    # Load model
    model = load_model(model_path, num_labels)

    # Prune the model
    pruned_model = prune_model(model, pruning_ratio=pruning_ratio)

    # Quantize the model
    quantized_model = quantize_model(pruned_model)

    # Validate the pruned and quantized model
    accuracy, val_loss = validate_model(quantized_model, dataset_name=dataset_name)

    # Save the pruned and quantized model
    save_model(quantized_model, output_path)

    logging.info(f"Final model accuracy after pruning and quantization: {accuracy:.4f}")

if __name__ == "__main__":
    main()
