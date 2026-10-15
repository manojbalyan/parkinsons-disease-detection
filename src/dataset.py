import json
from pathlib import Path
from typing import Tuple, List
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

class ParkinsonsDataset(Dataset):
    """
    Standard PyTorch Dataset wrapper.
    Converts numpy arrays into PyTorch Tensors for training.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Convert numbers to 32-bit float tensors (PyTorch standard)
        self.features = torch.tensor(features, dtype=torch.float32)
        # Unsqueeze adds an extra dimension so shape is (N, 1) matching model output
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        # Returns total number of patient samples
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns the 22 features and label for patient at index 'idx'
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data(
    data_path: str = "data/raw/parkinsons.data",
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Step 1: Loads the CSV file containing 195 patient voice measurements.
    Step 2: Drops patient ID column ('name').
    Step 3: Splits data into 80% Training and 20% Testing (keeping class ratio intact).
    Step 4: Normalizes features using StandardScaler (mean=0, std=1).
    Step 5: Saves the trained scaler and feature names list to models/ folder.
    """
    # 1. Read dataset CSV
    df = pd.read_csv(data_path)
    
    # 2. Drop the text ID column ('name' like phon_R01_S01_1)
    if "name" in df.columns:
        df = df.drop(columns=["name"])
        
    target_col = "status"  # 1 = Parkinson's, 0 = Healthy
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 3. Stratified Split (80% train = 156 samples, 20% test = 39 samples)
    # 'stratify=y' ensures train and test sets have the same ratio of Healthy vs Parkinson's
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # 4. Feature Scaling (Z-score normalization: (X - mean) / std)
    # We fit the scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Save scaler and feature names so app.py can use the exact same scaling
    if save_artifacts:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, models_dir / "scaler.pkl")
        with open(models_dir / "features.json", "w") as f:
            json.dump(feature_cols, f, indent=2)
            
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def get_dataloaders(
    data_path: str = "data/raw/parkinsons.data",
    batch_size: int = 16,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, StandardScaler, List[str]]:
    """
    Creates mini-batch DataLoaders for PyTorch training loop.
    Batch size = 16 (feeds 16 patient records at a time to the neural net).
    """
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        save_artifacts=True
    )
    
    train_dataset = ParkinsonsDataset(X_train, y_train)
    test_dataset = ParkinsonsDataset(X_test, y_test)
    
    # Shuffling train loader makes training more robust
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, feature_names

if __name__ == "__main__":
    train_loader, test_loader, scaler, feature_names = get_dataloaders()
    print("Dataset successfully loaded and preprocessed.")
    print(f"Total features: {len(feature_names)}")
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
