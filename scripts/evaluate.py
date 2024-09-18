import os
import yaml
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Prepare dataset
def tokenize_fn(tokenizer, examples, max_length=128):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

# Compute BLEU score for fluency and relevance
def compute_bleu_score(references, hypotheses):
    bleu_scores = [sentence_bleu([ref.split()], hyp.split()) for ref, hyp in zip(references, hypotheses)]
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

# Extended evaluation function with additional metrics
def evaluate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_texts = []
    all_pred_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Assuming text fields are available; replace with your actual fields
            texts = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            pred_texts = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            all_texts.extend(texts)
            all_pred_texts.extend(pred_texts)

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    bleu_score = compute_bleu_score(all_texts, all_pred_texts)

    return total_loss / len(dataloader), accuracy, precision, recall, f1, bleu_score

# Save evaluation results
def save_evaluation_results(results, output_file):
    with open(output_file, 'w') as f:
        yaml.dump(results, f)

# Main function
def main():
    # Load configurations
    config = load_config('config/evaluation_config.yaml')
    
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    num_labels = config['num_labels']
    batch_size = config['batch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_file = config.get('output_file', 'evaluation_results.yaml')

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_fn(tokenizer, x), batched=True)
    test_dataset = tokenized_dataset['test'].remove_columns(['text', 'label'])

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Load the model
    model_path = config.get('model_path', 'model_output/latest_model')
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluation
    eval_loss, accuracy, precision, recall, f1, bleu_score = evaluate(model, test_dataloader, criterion, device, tokenizer)

    # Print metrics
    print(f"Evaluation Loss: {eval_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"BLEU Score: {bleu_score}")

    # Save metrics to file
    evaluation_results = {
        'loss': eval_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'bleu_score': bleu_score
    }
    save_evaluation_results(evaluation_results, output_file)

if __name__ == "__main__":
    main()
