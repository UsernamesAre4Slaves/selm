import optuna
import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.model.transformer import TransformerModel  # Placeholder for custom model
from src.utils import load_data  # Placeholder utility for loading data
import yaml
from sklearn.metrics import accuracy_score

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Define the objective function for Optuna
def objective(trial):
    # Load configurations
    config = load_config('config/training_config.yaml')

    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 3, 10)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    num_labels = config['num_labels']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)
    
    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    train_dataset = tokenized_dataset['train']
    test_dataset = tokenized_dataset['test']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy

def main():
    study = optuna.create_study(direction='maximize')  # We want to maximize accuracy
    study.optimize(objective, n_trials=10)  # Number of trials for optimization

    print("Best hyperparameters found:")
    print(study.best_params)
    print(f"Best accuracy: {study.best_value}")

if __name__ == "__main__":
    main()
