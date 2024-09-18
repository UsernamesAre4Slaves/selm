import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from src.model.transformer import Transformer
from src.data.load_data import load_qa_data
from src.utils.utils import save_model, load_model
from src.knowledge_graph.gnn import EnhancedGNN
import logging
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QADataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_length):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]

        # Tokenize and pad the inputs
        question_encodings = self.tokenizer(question, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        context_encodings = self.tokenizer(context, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        answer_encoding = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'question': question_encodings['input_ids'].squeeze(0),
            'context': context_encodings['input_ids'].squeeze(0),
            'answer': answer_encoding['input_ids'].squeeze(0)
        }

def train(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        questions = batch['question'].to(device)
        contexts = batch['context'].to(device)
        answers = batch['answer'].to(device)

        optimizer.zero_grad()
        
        with autocast():
            outputs = model(questions, contexts, None, None)
            loss = criterion(outputs, answers)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            questions = batch['question'].to(device)
            contexts = batch['context'].to(device)
            answers = batch['answer'].to(device)

            outputs = model(questions, contexts, None, None)
            loss = criterion(outputs, answers)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(answers.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return avg_loss, accuracy, f1

def integrate_knowledge(model, gnn_model, question, context, device):
    # Tokenize question and context
    question_encodings = tokenizer(question, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    context_encodings = tokenizer(context, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    question_ids = question_encodings['input_ids'].to(device)
    context_ids = context_encodings['input_ids'].to(device)

    # Get model outputs
    with torch.no_grad():
        answer_logits = model(question_ids, context_ids, None, None)
        answer_predictions = torch.argmax(answer_logits, dim=1)

        # Use the GNN model for knowledge-based reasoning
        gnn_data = prepare_gnn_input(question, context)
        gnn_output = gnn_model(gnn_data['x'], gnn_data['edge_index'], gnn_data['batch'])
        knowledge_based_answer = torch.argmax(gnn_output, dim=1)

    return knowledge_based_answer

def prepare_gnn_input(question, context):
    # Prepare GNN data based on question and context
    # This is a placeholder function. You should replace it with actual logic.
    x = torch.randn((100, 16))  # Example node features
    edge_index = torch.randint(0, 100, (2, 200), dtype=torch.long)  # Example edge indices
    batch = torch.zeros(100, dtype=torch.long)  # Example batch assignment

    return {
        'x': x,
        'edge_index': edge_index,
        'batch': batch
    }

def main():
    # Hyperparameters and configuration
    embed_size = 256
    num_layers = 4
    heads = 8
    dropout = 0.1
    forward_expansion = 4
    num_classes = 10  # Number of possible answers
    batch_size = 32
    num_epochs = 5
    learning_rate = 1e-4
    max_length = 512  # Max length for tokenization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    tokenizer = ...  # Initialize your tokenizer here
    train_questions, train_contexts, train_answers = load_qa_data('train')
    val_questions, val_contexts, val_answers = load_qa_data('val')

    train_dataset = QADataset(train_questions, train_contexts, train_answers, tokenizer, max_length)
    val_dataset = QADataset(val_questions, val_contexts, val_answers, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = Transformer(embed_size, num_layers, heads, dropout, forward_expansion, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Initialize GNN model for knowledge graph reasoning
    gnn_model = EnhancedGNN(in_channels=16, hidden_channels=32, out_channels=num_classes, dropout=0.5).to(device)

    # Training and evaluation loop
    best_val_loss = float('inf')
    patience = 2
    trigger_times = 0
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, criterion, device)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}')
        logger.info(f'Val Accuracy: {val_accuracy:.4f}')
        logger.info(f'Val F1 Score: {val_f1:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best model
            save_model(model, 'best_qa_transformer_model.pth')
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logger.info('Early stopping triggered')
                break

if __name__ == "__main__":
    main()
