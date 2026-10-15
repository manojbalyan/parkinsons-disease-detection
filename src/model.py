import torch
import torch.nn as nn

class ParkinsonNet(nn.Module):
    """
    Simple 4-Layer Deep Neural Network (MLP) for Parkinson's Disease Detection.
    
    Structure:
      - Input Layer: 22 voice features (Pitch, Jitter, Shimmer, PPE, etc.)
      - Hidden Layer 1: 64 neurons
      - Hidden Layer 2: 32 neurons
      - Hidden Layer 3: 16 neurons
      - Output Layer: 1 neuron (outputs probability: 0 = Healthy, 1 = Parkinson's)
    """
    def __init__(self, input_dim: int = 22, hidden_dim1: int = 64, hidden_dim2: int = 32, hidden_dim3: int = 16, dropout_rate: float = 0.3):
        super(ParkinsonNet, self).__init__()
        
        # --- Layer 1: 22 inputs -> 64 neurons ---
        self.fc1 = nn.Linear(input_dim, hidden_dim1)       # Linear connection: inputs multiplied by weights + bias
        self.bn1 = nn.BatchNorm1d(hidden_dim1)             # Batch Normalization: balances numbers so training is smooth
        self.act1 = nn.LeakyReLU(negative_slope=0.1)       # Activation: turns neuron ON; slope 0.1 prevents dead neurons
        self.drop1 = nn.Dropout(p=dropout_rate)            # Dropout: turns off 30% neurons randomly to stop overfitting
        
        # --- Layer 2: 64 neurons -> 32 neurons ---
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)     # Shrinks down to 32 neurons to combine features
        self.bn2 = nn.BatchNorm1d(hidden_dim2)             # Batch Normalization
        self.act2 = nn.LeakyReLU(negative_slope=0.1)       # Activation
        self.drop2 = nn.Dropout(p=dropout_rate * 0.7)      # Slightly gentler dropout (~21%)
        
        # --- Layer 3: 32 neurons -> 16 neurons ---
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)     # Shrinks to 16 neurons to summarize patterns
        self.act3 = nn.LeakyReLU(negative_slope=0.1)       # Activation
        
        # --- Output Layer: 16 neurons -> 1 output ---
        self.out = nn.Linear(hidden_dim3, 1)               # Single output score (logit)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Passes the 22 numbers through the layers one by one.
        - If return_logits is True: returns raw score (used during training with BCEWithLogitsLoss).
        - If return_logits is False: applies Sigmoid to convert score into a 0 to 1 probability.
        """
        # Pass through Layer 1
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        
        # Pass through Layer 2
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        
        # Pass through Layer 3
        x = self.act3(self.fc3(x))
        
        # Output score
        logits = self.out(x)
        
        if return_logits:
            return logits
        
        # Sigmoid squashes output into [0, 1] probability
        return torch.sigmoid(logits)

if __name__ == "__main__":
    # Quick sanity test
    model = ParkinsonNet(input_dim=22)
    dummy_input = torch.randn(4, 22)  # 4 fake patients with 22 features each
    output = model(dummy_input)
    print("Model Architecture:\n", model)
    print("\nSample Output Probabilities (0 to 1):\n", output)
