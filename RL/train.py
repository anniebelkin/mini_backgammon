import datetime
import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from RL.game import sample_random

SAMPLES_FILE = "RL/board_samples"
RANDOM_SAMPLES_FILE = "RL/board_random_samples"
PRETRAIN_OUTPUT = "RL/pretrain_output"

run_number = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(PRETRAIN_OUTPUT, f"RUN_{run_number}")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving outputs to directory: {output_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HeuristicNN(nn.Module):
    def __init__(self, input_size=48, hidden_size=64):
        super(HeuristicNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_random_samples(file_path):
    """Load random board samples from file. Returns X (features) and y (heuristic targets)."""
    X, y = [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                X.append(record["board_vector"])
                y.append(record["heuristic"])
    return X, y

def pretrain(learning_rate=0.001, batch_update_size=50, games=10000, 
             improvement_threshold=0.0001, patience=100, should_sample=False):
    """
    Pretrains the HeuristicNN model using random board samples.
    
    Steps:
      1. Generate random samples (if not already present).
      2. Load and shuffle data.
      3. Train in mini-batches using MSELoss (sum reduction).
      4. Plot the sum of MSE loss per epoch and apply early stopping.
    """
    # Generate random samples.
    if should_sample:
        sample_random(games=games, sample_file_name=RANDOM_SAMPLES_FILE)
    
    # Load data.
    X, y = load_random_samples(RANDOM_SAMPLES_FILE)
    if not X:
        print("No samples found. Exiting.")
        return
    
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    dataset_size = len(X)
    print(f"[PRETRAIN] Loaded {dataset_size} samples.")

    # Initialize model, optimizer, and criterion.
    model = HeuristicNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')
    
    epoch_losses = []
    last_epoch_loss = None
    no_improvement_count = 0
    epoch_count = 0
    
    # Training loop with early stopping.
    while True:
        epoch_count += 1
        model.train()
        
        # Shuffle data.
        indices = list(range(dataset_size))
        random.shuffle(indices)
        X_tensor = X_tensor[indices]
        y_tensor = y_tensor[indices]
        
        epoch_loss = 0.0
        idx = 0
        while idx < dataset_size:
            batch_end = min(idx + batch_update_size, dataset_size)
            batch_X = X_tensor[idx:batch_end]
            batch_y = y_tensor[idx:batch_end]
            
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            idx = batch_end
        
        epoch_losses.append(epoch_loss)
        print(f"[Epoch {epoch_count}] Sum of MSE Loss: {epoch_loss:.6f}")
        
        # Plot loss curve.
        plt.clf()
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
        plt.title("Pretraining Loss Over Epochs (Early Stopping)")
        plt.xlabel("Epoch")
        plt.ylabel("Sum of MSE Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "pretrain_loss.png"))
        
        # Early stopping check.
        if last_epoch_loss is not None:
            improvement = abs(last_epoch_loss - epoch_loss)
            if improvement < improvement_threshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
        last_epoch_loss = epoch_loss
        
        if no_improvement_count >= patience:
            print(f"No significant improvement for {patience} epochs. Early stopping.")
            break
    
    model_path = os.path.join(output_dir, "pretrain_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Training ended after {epoch_count} epochs. Model saved to {model_path}")
