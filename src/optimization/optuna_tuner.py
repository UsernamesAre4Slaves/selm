import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.model.transformer import Transformer
from src.data.load_data import load_data
from src.utils.utils import save_model
from sklearn.metrics import accuracy_score, f1_score

def objective(trial):
    # Define hyperparameters to optimize
    embed_size = trial.suggest_categorical('embed_size', [128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    heads = trial.suggest_int('heads', 4, 8)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    forward_expansion = trial.suggest_int('forward_expansion', 2, 8)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    num_classes = 10  # Example value, adjust as needed

    # Load data
    train_loader, val_loader = load_data(batch_size=32)  # Adjust batch size and data loading as needed

    # Initialize model, loss function, and optimizer
    model = Transformer(embed_size, num_layers, heads, dropout, forward_expansion, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            src, tgt, labels = batch
            optimizer.zero_grad()
            output = model(src, tgt, None, None)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                src, tgt, labels = batch
                output = model(src, tgt, None, None)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model checkpoint
            checkpoint_path = f'best_model_epoch_{epoch}.pth'
            save_model(model, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    return val_loss

def main():
    # Create Optuna study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)  # Adjust number of trials as needed

    # Print best parameters and save best model
    print('Best parameters:', study.best_params)
    print('Best value:', study.best_value)

    # Save the best model with the best hyperparameters
    best_params = study.best_params
    best_model = Transformer(
        embed_size=best_params['embed_size'],
        num_layers=best_params['num_layers'],
        heads=best_params['heads'],
        dropout=best_params['dropout'],
        forward_expansion=best_params['forward_expansion'],
        num_classes=10  # Example value, adjust as needed
    )
    
    # Load best model weights if available
    save_model(best_model, 'best_transformer_model.pth')

if __name__ == "__main__":
    main()
