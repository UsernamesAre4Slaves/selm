import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.model.transformer import Transformer
from src.data.load_data import load_active_learning_data
from src.utils.utils import save_model, load_model
from sklearn.metrics import uncertainty_score
import torch.nn as nn
import torch.optim as optim

class ActiveLearningDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and pad the input
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'text': encoding['input_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(texts, texts, None, None)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            outputs = model(texts, texts, None, None)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def uncertainty_sampling(model, dataloader, device, n_samples):
    model.eval()
    uncertainties = []
    indices = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            texts = batch['text'].to(device)

            outputs = model(texts, texts, None, None)
            probs = torch.softmax(outputs, dim=1)
            uncertainty = 1 - torch.max(probs, dim=1).values
            uncertainties.extend(uncertainty.cpu().numpy())
            indices.extend([i] * len(uncertainty))

    uncertainties = np.array(uncertainties)
    indices = np.array(indices)
    uncertain_indices = np.argsort(uncertainties)[-n_samples:]
    return uncertain_indices

def main():
    # Hyperparameters and configuration
    embed_size = 256
    num_layers = 4
    heads = 8
    dropout = 0.1
    forward_expansion = 4
    num_classes = 10  # Number of classes
    batch_size = 32
    num_epochs = 5
    learning_rate = 1e-4
    n_samples = 100  # Number of samples to select for active learning
    max_length = 512  # Max length for tokenization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    tokenizer = ...  # Initialize your tokenizer here
    all_texts, all_labels = load_active_learning_data('all')
    initial_texts, initial_labels = load_active_learning_data('initial')

    # Initialize datasets
    initial_dataset = ActiveLearningDataset(initial_texts, initial_labels, tokenizer, max_length)
    all_dataset = ActiveLearningDataset(all_texts, all_labels, tokenizer, max_length)
    
    # Create DataLoader
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = Transformer(embed_size, num_layers, heads, dropout, forward_expansion, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initial training
    initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, initial_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, initial_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Active learning loop
    for iteration in range(5):  # Number of active learning iterations
        print(f"Active Learning Iteration {iteration + 1}")

        uncertain_indices = uncertainty_sampling(model, all_loader, device, n_samples)
        new_texts = [all_texts[i] for i in uncertain_indices]
        new_labels = [all_labels[i] for i in uncertain_indices]

        # Append to initial dataset
        initial_texts.extend(new_texts)
        initial_labels.extend(new_labels)
        initial_dataset = ActiveLearningDataset(initial_texts, initial_labels, tokenizer, max_length)
        initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)

        # Train on the updated dataset
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train(model, initial_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, initial_loader, criterion, device)
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Save the trained model
    save_model(model, 'active_learning_transformer_model.pth')

if __name__ == "__main__":
    main()
