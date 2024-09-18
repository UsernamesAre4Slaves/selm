import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import wandb  # For logging and tracking
import sys

# Initialize Weights & Biases for logging
wandb.init(project="chatbot_specific_training")

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Training function
def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        
        optimizer.zero_grad()
        with autocast():  # Mixed Precision Training
            outputs = model(inputs, labels=labels)
            loss = criterion(outputs.logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)

            with autocast():  # Mixed Precision Training
                outputs = model(inputs, labels=labels)
                loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy

# Main function
def main():
    # Determine if this is chatbot-specific training from the command line argument
    chatbot_specific = '--chatbot' in sys.argv

    # Load configurations
    config = load_config('config/training_config.yaml')
    
    model_name = config['model_name']
    dataset_name = config['chatbot_dataset'] if chatbot_specific else config['dataset_name']
    num_labels = config['num_labels']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    warmup_steps = config.get('warmup_steps', 0)
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

    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * num_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # For mixed precision training

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train(model, train_dataloader, optimizer, criterion, device, scaler)
        eval_loss, accuracy = evaluate(model, test_dataloader, criterion, device)
        
        print(f"Training loss: {train_loss}")
        print(f"Evaluation loss: {eval_loss}, Accuracy: {accuracy}")

        # Log metrics to Weights & Biases
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'accuracy': accuracy
        })

        # Save model checkpoint
        output_dir = f"model_output/chatbot_epoch_{epoch + 1}" if chatbot_specific else f"model_output/epoch_{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    main()
