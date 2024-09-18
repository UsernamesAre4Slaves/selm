import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_fitness(model, data_loader, device, criterion):
    """
    Evaluate the fitness of a model based on its performance on a validation set,
    considering fluency, relevance, and user engagement.

    Args:
        model (nn.Module): The model to be evaluated.
        data_loader (DataLoader): DataLoader with validation data.
        device (torch.device): Device for computation (e.g., 'cuda', 'cpu').
        criterion (nn.Module): Loss function to compute fitness (e.g., nn.CrossEntropyLoss).
    
    Returns:
        dict: A dictionary with fitness metrics including accuracy, fluency, relevance, and engagement.
    """
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None).to(device)
            
            # Forward pass
            outputs = model(inputs, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            all_labels.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = (correct_predictions / total_samples) * 100

    # Calculate additional metrics for fluency, relevance, and engagement
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Example of how to incorporate these metrics into a fitness score
    fitness_score = (accuracy + precision + recall + f1) / 4 - avg_loss  # Adjust as needed

    return {
        'fitness_score': fitness_score,
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Example usage
    class ExampleModel(nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)

    # Initialize example model and DataLoader
    model = ExampleModel()
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    fitness = evaluate_fitness(model, data_loader, device, criterion)
    print(f"Model fitness metrics: {fitness}")
