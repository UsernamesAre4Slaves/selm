import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicInferenceModel(nn.Module):
    def __init__(self, transformer, exit_threshold=0.95, num_exits=3):
        """
        Dynamic inference model with early exits for conditional computation.
        
        Args:
            transformer (nn.Module): The underlying transformer model.
            exit_threshold (float): Confidence threshold for early exit.
            num_exits (int): Number of early exits allowed during inference.
        """
        super(DynamicInferenceModel, self).__init__()
        self.transformer = transformer
        self.exit_threshold = exit_threshold
        self.num_exits = num_exits
        
        # Define early exit heads for intermediate layers
        self.exit_heads = nn.ModuleList(
            [nn.Linear(self.transformer.hidden_size, self.transformer.num_labels) for _ in range(num_exits)]
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with dynamic inference and early exits.
        
        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor, optional): Attention mask.
        
        Returns:
            Tuple: The final prediction and the intermediate exits (if taken).
        """
        hidden_states = self.transformer(input_ids, attention_mask)
        final_output = hidden_states[-1]
        early_exit_taken = False
        
        # Iterate over hidden states for early exits
        for i, hidden_state in enumerate(hidden_states):
            if i >= self.num_exits:
                break

            # Compute intermediate logits
            logits = self.exit_heads[i](hidden_state)
            confidence = torch.max(F.softmax(logits, dim=-1), dim=-1)[0]

            # Check confidence threshold for early exit
            if confidence.mean().item() > self.exit_threshold:
                early_exit_taken = True
                return logits, early_exit_taken

        # If no early exit is taken, return final output
        final_logits = self.exit_heads[-1](final_output)
        return final_logits, early_exit_taken

def dynamic_inference(model, input_data, attention_mask=None, exit_threshold=0.95):
    """
    Perform inference with dynamic early exits.
    
    Args:
        model (DynamicInferenceModel): The dynamic inference model.
        input_data (Tensor): The input data to be processed.
        attention_mask (Tensor, optional): Attention mask for the input data.
        exit_threshold (float): Confidence threshold for early exits.
    
    Returns:
        Tensor: Model prediction, possibly from an early exit.
    """
    logits, early_exit = model(input_data, attention_mask)
    
    if early_exit:
        print("Early exit taken based on confidence.")
    else:
        print("Full model inference completed.")
    
    return logits

def measure_efficiency(model, data_loader, device):
    """
    Measure the efficiency of dynamic inference in terms of computation saved.
    
    Args:
        model (DynamicInferenceModel): The dynamic inference model.
        data_loader (DataLoader): DataLoader with test data.
        device (torch.device): Device for inference (e.g., 'cuda', 'cpu').
    
    Returns:
        float: Percentage of early exits taken across the dataset.
    """
    early_exit_count = 0
    total_count = 0

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None).to(device)
            
            # Perform inference and count early exits
            _, early_exit = model(input_ids, attention_mask)
            if early_exit:
                early_exit_count += 1
            
            total_count += 1
    
    return (early_exit_count / total_count) * 100

if __name__ == "__main__":
    # Example usage of dynamic inference

    # Placeholder transformer model (replace with actual model)
    class ExampleTransformer(nn.Module):
        def __init__(self, hidden_size=768, num_labels=2):
            super(ExampleTransformer, self).__init__()
            self.hidden_size = hidden_size
            self.num_labels = num_labels
            self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(12)])
            self.output_layer = nn.Linear(hidden_size, num_labels)
        
        def forward(self, input_ids, attention_mask=None):
            hidden_states = []
            x = input_ids.float()  # Placeholder for actual embedding lookup
            for layer in self.layers:
                x = layer(x)
                hidden_states.append(x)
            return hidden_states

    # Initialize model
    transformer_model = ExampleTransformer()
    dynamic_model = DynamicInferenceModel(transformer=transformer_model, exit_threshold=0.95, num_exits=3)

    # Example input data (replace with actual data)
    input_data = torch.randint(0, 1000, (1, 128))  # Simulating input token IDs

    # Perform dynamic inference
    output = dynamic_inference(dynamic_model, input_data)
    print("Inference output:", output)
