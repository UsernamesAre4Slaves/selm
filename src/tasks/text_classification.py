import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler
from src.model.transformer import Transformer
from src.data.load_data import load_text_classification_data
from src.utils.utils import save_model, load_model
from sklearn.metrics import accuracy_score

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Remove batch dimension
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with autocast():  # Mixed Precision Training
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():  # Mixed Precision Training
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels

def predict(model, text, tokenizer, device):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

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
    warmup_steps = 1000  # Example value, adjust as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_texts, train_labels = load_text_classification_data('train')
    val_texts, val_labels = load_text_classification_data('val')

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = Transformer(embed_size, num_layers, heads, dropout, forward_expansion, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * num_epochs)
    scaler = GradScaler()  # For mixed precision training

    # Training and evaluation loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_accuracy, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Save model checkpoint
        output_dir = f"model_output/epoch_{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)
        save_model(model, os.path.join(output_dir, 'text_classification_transformer_model.pth'))

        # Log metrics to Weights & Biases
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

        scheduler.step()

    print("Training complete!")
    wandb.finish()

    # Example inference
    sample_text = "Example text for classification."
    label = predict(model, sample_text, tokenizer, device)
    print(f"Predicted Label: {label}")

if __name__ == "__main__":
    main()
