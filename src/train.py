import json
import sys
from pathlib import Path

# Add project root to sys.path so imports work cleanly from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from src.dataset import get_dataloaders
from src.model import ParkinsonNet

def set_seed(seed: int = 42):
    """Sets random seeds so training results are reproducible every time."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(
    data_path: str = "data/raw/parkinsons.data",
    epochs: int = 120,
    batch_size: int = 16,
    learning_rate: float = 0.002,
    seed: int = 42
):
    """
    Main training function:
    1. Loads train & validation data batches (DataLoader).
    2. Builds the 4-layer ParkinsonNet model.
    3. Uses Binary Cross-Entropy Loss (BCEWithLogitsLoss) and AdamW Optimizer.
    4. Trains over 120 epochs, tracking loss and accuracy.
    5. Saves the checkpoint with the highest validation accuracy to models/.
    6. Generates loss and accuracy plots in reports/.
    """
    set_seed(seed)
    
    # 1. Load Data
    train_loader, test_loader, scaler, feature_names = get_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        test_size=0.2,
        random_state=seed
    )
    
    # Pick Apple Silicon MPS if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # 2. Initialize Model
    model = ParkinsonNet(input_dim=len(feature_names)).to(device)
    
    # 3. Loss & Optimizer
    # BCEWithLogitsLoss combines Sigmoid + Binary Cross-Entropy in a numerically stable way
    criterion = nn.BCEWithLogitsLoss()
    # AdamW adjusts learning rate dynamically per parameter and includes weight decay (L2 penalty)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Reduces learning rate if validation loss stops improving for 8 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_parkinsons_model.pth"
    
    print("\n--- Starting Neural Network Training ---")
    
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()                               # Reset gradients from previous step
            logits = model(batch_features, return_logits=True)  # Forward pass (get model prediction)
            loss = criterion(logits, batch_labels)              # Calculate error/loss
            loss.backward()                                     # Backward pass (calculate gradients)
            optimizer.step()                                    # Update neuron weights
            
            running_train_loss += loss.item() * len(batch_labels)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():  # Turn off gradient calculation during testing
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(batch_features, return_logits=True)
                loss = criterion(logits, batch_labels)
                running_val_loss += loss.item() * len(batch_labels)
                
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int().cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(batch_labels.cpu().numpy().flatten())
                
        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        epoch_val_acc = accuracy_score(all_targets, all_preds)
        scheduler.step(epoch_val_loss)
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        # Save the model whenever validation loss reaches a new best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "input_dim": len(feature_names),
                "feature_names": feature_names
            }, best_model_path)
            
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")
            
    print("\nTraining Complete!")
    print(f"Best Model Saved to: {best_model_path}")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Validation Accuracy: {best_val_acc * 100:.2f}%\n")
    
    # Save training history JSON
    with open(reports_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    # Plot learning curves
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="crimson", linestyle="--")
    plt.title("Loss Curves (BCE Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in history["val_acc"]], label="Val Accuracy (%)", color="forestgreen")
    plt.title("Validation Accuracy Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(reports_dir / "learning_curves.png", dpi=200)
    plt.close()
    print(f"Saved learning curves plot to: {reports_dir / 'learning_curves.png'}")

if __name__ == "__main__":
    train_model()
