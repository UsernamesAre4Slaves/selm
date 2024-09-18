import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from transformers import BartTokenizer, BartForConditionalGeneration
from src.data.load_data import load_summarization_data
from src.utils.utils import save_model

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        summary_encoding = self.tokenizer(summary, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': summary_encoding['input_ids'].squeeze()
        }

        return item

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def generate_summary(model, tokenizer, text, device, max_length=512):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length).to(device)
        summary_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

def main():
    # Hyperparameters and configuration
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 32
    num_epochs = 5
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_texts, train_summaries = load_summarization_data('train')
    val_texts, val_summaries = load_summarization_data('val')

    train_dataset = SummarizationDataset(train_texts, train_summaries, tokenizer)
    val_dataset = SummarizationDataset(val_texts, val_summaries, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    # Training and evaluation loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')

    # Save the trained model
    save_model(model, 'summarization_transformer_model.pth')

    # Example inference
    sample_text = "Example text to summarize."
    summary = generate_summary(model, tokenizer, sample_text, device)
    print(f"Generated Summary: {summary}")

if __name__ == "__main__":
    main()
