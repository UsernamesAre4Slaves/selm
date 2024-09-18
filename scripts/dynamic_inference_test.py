import torch
import torch.nn.functional as F
import argparse
from src.model.transformer import SELMTransformer
from src.tasks.text_classification import TextClassificationDataset
from torch.utils.data import DataLoader
import yaml

def load_model(config):
    """Load the SELM model with the given configuration."""
    model = SELMTransformer(config_path=config['model_config_path'])
    
    # Load checkpoint if provided
    if config['dynamic_inference']['checkpoint_path']:
        checkpoint = torch.load(config['dynamic_inference']['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {config['dynamic_inference']['checkpoint_path']}")
    
    return model

def dynamic_inference(model, data_loader, config):
    """Perform dynamic inference with early exits based on confidence thresholds."""
    model.eval()
    
    results = []
    confidence_threshold = config['dynamic_inference']['confidence_threshold']
    max_layers = config['dynamic_inference']['max_layers']
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(config['device'])
            
            # Perform dynamic inference with early exits
            for layer_num, output in enumerate(model.iterate_layers(inputs), start=1):
                logits = model.output_layer(output)
                probs = F.softmax(logits, dim=-1)
                max_conf, predicted_class = torch.max(probs, dim=-1)

                # Check if the confidence exceeds the threshold
                if torch.any(max_conf > confidence_threshold):
                    # Exit early if the confidence is high enough
                    early_exit = max_conf > confidence_threshold
                    results.extend(predicted_class[early_exit].cpu().tolist())
                    break
                
                # If max layers are reached, exit without meeting the confidence threshold
                if layer_num == max_layers:
                    results.extend(predicted_class.cpu().tolist())
                    break

    return results

def main():
    parser = argparse.ArgumentParser(description='Dynamic Inference with Early Exits for SELM Model')
    parser.add_argument('--config', type=str, default='config/dynamic_inference_config.yaml', help='Path to the config file.')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(config)
    model = model.to(config['device'])

    # Load dataset
    dataset = TextClassificationDataset(config['data']['test_file'])
    data_loader = DataLoader(dataset, batch_size=config['inference']['batch_size'], shuffle=False)

    # Run dynamic inference
    results = dynamic_inference(model, data_loader, config)

    # Save results
    with open(config['dynamic_inference']['output_file'], 'w') as f:
        for result in results:
            f.write(f"{result}\n")

if __name__ == '__main__':
    main()
