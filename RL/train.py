import datetime
import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from RL.game import sample_random, sample_best_of_four, test_model_against_heuristic, test_model_improvement, sample_best_of_four_hard_huristic, test_model_against_hard_heuristic, test_model_against_player

# File paths and constants
SAMPLES_FILE = "RL/board_samples"
RANDOM_SAMPLES_FILE = "RL/board_random_samples"
PRETRAIN_OUTPUT = "RL/pretrain_output"
TRAIN_OUTPUT = "RL/train_output"
SAVED_PRETRAINED_MODEL = "pretrained_model.pt"
SAVED_TRAINED_MODEL = "trained_model.pt"
RL_DIR = "RL"

run_number = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
pretrained_output_dir = os.path.join(PRETRAIN_OUTPUT, f"RUN_{run_number}")
output_dir = os.path.join(TRAIN_OUTPUT, f"RUN_{run_number}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CPU for debugging/testing (remove this if you want GPU)
device = torch.device("cpu")
print(f"Using device: {device}")

def trim_samples(file_path, max_entries=20000):
    with open(file_path, "r") as f:
        lines = f.readlines()
    if len(lines) > max_entries:
        # Keep only the last max_entries lines.
        lines = lines[-max_entries:]
        with open(file_path, "w") as f:
            f.writelines(lines)

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
        # Ensure output is between 0 and 1
        return torch.sigmoid(x)

def load_samples(file_path, target="target"):
    """Load board samples from file. Returns X (features) and y (target values)."""
    X, y = [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                X.append(record["board_vector"])
                y.append(record[target])
    return X, y

def load_model(model_path, backup_path):
    """Load the pretrained model from model_path. If not found, load from backup."""
    model = HeuristicNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Failed to load model from {model_path}. Loading backup from {backup_path}")
        model.load_state_dict(torch.load(backup_path, map_location=device))
    return model

def pretrain(learning_rate=0.001, batch_update_size=50, games=5000, 
             improvement_threshold=0.0001, patience=100, should_sample=False):
    """
    Pretrains the HeuristicNN model using random board samples.
    
    Steps:
      1. Generate random samples (if needed).
      2. Load and shuffle data.
      3. Train in mini-batches using MSELoss (sum reduction).
      4. Plot the sum of MSE loss per epoch and apply early stopping.
    """
    os.makedirs(pretrained_output_dir, exist_ok=True)
    print(f"Saving pre-training outputs to: {pretrained_output_dir}")

    if should_sample:
        sample_random(games=games, sample_file_name=RANDOM_SAMPLES_FILE)
    
    X, y = load_samples(RANDOM_SAMPLES_FILE, target="heuristic")
    if not X:
        print("No samples found. Exiting.")
        return

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    dataset_size = len(X)
    print(f"[PRETRAIN] Loaded {dataset_size} samples.")

    model = HeuristicNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')
    
    epoch_losses = []
    last_epoch_loss = None
    no_improvement_count = 0
    epoch_count = 0

    try:
        while True:
            epoch_count += 1
            model.train()

            # Shuffle the data
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

            print(f"[Epoch {epoch_count}] Sum of MSE Loss: {epoch_loss:.6f}")
            if epoch_count > 1:
                epoch_losses.append(epoch_loss)
                plt.clf()
                plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
                plt.title("Pretraining Loss Over Epochs (Early Stopping)")
                plt.xlabel("Epoch")
                plt.ylabel("Sum of MSE Loss")
                plt.grid(True)
                plt.savefig(os.path.join(pretrained_output_dir, "pretrain_loss.png"))
            
            if last_epoch_loss is not None:
                improvement = abs(last_epoch_loss - epoch_loss)
                if improvement < improvement_threshold:
                    no_improvement_count += 1
                else:
                    print(f"last imorovement count: {no_improvement_count}")
                    no_improvement_count = 0
            last_epoch_loss = epoch_loss
            
            if no_improvement_count >= patience:
                print(f"No significant improvement for {patience} epochs. Early stopping.")
                break
    finally:
        model_backup_path = os.path.join(pretrained_output_dir, SAVED_PRETRAINED_MODEL)
        model_path = os.path.join(RL_DIR, SAVED_PRETRAINED_MODEL)
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), model_backup_path)
        print(f"Training ended after {epoch_count} epochs. Model saved to {model_path} and {model_backup_path}")

def training(learning_rate=0.001, batch_update_size=50, epochs_per_round=5,
             max_rounds=50, test_games=10, patience=10, improvement_threshold=0.01,
             desired_win_rate=1.0):
    """
    Outcome-based training:
      1. Load pretrained model from RL_DIR/SAVED_PRETRAINED_MODEL.
      2. Backup current model in output_dir.
      3. In each round:
         - Sample new boards using best-of-four strategy.
         - Train on these samples for a fixed number of epochs.
         - Test the model (10 games) with test_model functions.
         - Update the model if performance improves.
         - Plot average wins vs. training round.
         - Also plot the improvement (new vs. old network win rate) over rounds.
      4. Continue until average wins reach the desired win rate.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving training outputs to: {output_dir}")

    final_path = os.path.join(output_dir, SAVED_TRAINED_MODEL)
    final_rl_path = os.path.join(RL_DIR, SAVED_TRAINED_MODEL)
    pretrained_path = os.path.join(RL_DIR, SAVED_PRETRAINED_MODEL)
    model = load_model(final_path, pretrained_path)

    backup_path = os.path.join(output_dir, "backup_pretrained.pt")
    torch.save(model.state_dict(), backup_path)
    print(f"Backed up pretrained model to {backup_path}")

    avg_wins_history = []
    improvement_history = []
    best_avg_wins = 0.0
    training_count = 0

    # Use default MSELoss (mean reduction) for outcome-based training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Keep a copy of the last model for improvement evaluation.
    last_model = HeuristicNN().to(device)
    last_model.load_state_dict(model.state_dict())

    try:
        while best_avg_wins < desired_win_rate and training_count < max_rounds:
            training_count += 1
            print(f"=== Training Round {training_count}/{max_rounds} ===")

            # Remove old sample file if exists.
            if os.path.exists(SAMPLES_FILE):
                os.remove(SAMPLES_FILE)
            # Generate new samples using best-of-four strategy.
            sample_best_of_four_hard_huristic(games=100, sample_file_name=SAMPLES_FILE, model=model, device=device)
            trim_samples(SAMPLES_FILE)

            X, y = load_samples(SAMPLES_FILE)
            if not X:
                print("No new samples found. Exiting.")
                break

            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)
            dataset_size = len(X_tensor)
            print(f"[Round {training_count}] Loaded {dataset_size} outcome-based samples.")

            last_epoch_loss = None
            no_improvement_count = 0

            # Train for a fixed number of epochs in this round.
            model.train()
            for ep in range(1, epochs_per_round + 1):
                indices = list(range(dataset_size))
                random.shuffle(indices)
                X_tensor = X_tensor[indices]
                y_tensor = y_tensor[indices]

                epoch_loss = 0.0
                idx = 0
                while idx < dataset_size:
                    end = min(idx + batch_update_size, dataset_size)
                    batch_X = X_tensor[idx:end]
                    batch_y = y_tensor[idx:end]

                    optimizer.zero_grad()
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    idx = end

                print(f"    Epoch {ep}/{epochs_per_round}, MSE Loss: {epoch_loss:.4f}")

                if last_epoch_loss is not None:
                    improvement_val = abs(last_epoch_loss - epoch_loss)
                    if improvement_val < improvement_threshold:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                last_epoch_loss = epoch_loss

                if no_improvement_count >= patience:
                    print(f"No significant improvement for {patience} epochs. Early stopping this round.")
                    break

            # Test the model against the heuristic opponent.
            model.eval()
            avg_wins = test_model_against_hard_heuristic(device, model)
            avg_wins_history.append(avg_wins)
            if avg_wins > best_avg_wins:
                best_avg_wins = avg_wins
            # Also test improvement against the last saved model.
            improv = test_model_improvement(device, model, last_model)
            improvement_history.append(improv)
            if improv > 0.5 or avg_wins >= best_avg_wins:
                last_model.load_state_dict(model.state_dict())
                torch.save(model.state_dict(), final_rl_path)
                torch.save(model.state_dict(), final_path)
            
            print(f"    Testing after round {training_count}: avg wins = {avg_wins:.3f}, improvement = {(improv-0.5):.3f}")

            # Plot progress of average wins.
            plt.clf()
            plt.plot(range(1, len(avg_wins_history) + 1), avg_wins_history, marker='o')
            plt.title("Outcome-Based Training Progress (Win % vs. Heuristic)")
            plt.xlabel("Round")
            plt.ylabel("Average Wins (10 games)")
            plt.ylim([0, 1.05])
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "training_progress.png"))
            
            # Plot progress of improvement (new vs. old network win %).
            plt.clf()
            plt.plot(range(1, len(improvement_history) + 1), improvement_history, marker='o', color='orange')
            plt.title("Improvement: New vs. Old Network")
            plt.xlabel("Round")
            plt.ylabel("Win % (10 games)")
            plt.ylim([0, 1.05])
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "improvement_progress.png"))

            if avg_wins >= desired_win_rate:
                print("Reached desired win rate. Stopping training.")
                break
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Finished training.")

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("RL/trained_model.pt", "").to(device)
    wins = 0
    model.eval()
    for _ in range(100):
        avg_wins = test_model_against_heuristic(device, model)
        wins += avg_wins * 10
    print(f"Model wins: {wins}/1000")

def test_specific_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, "").to(device)
    model.eval()
    wins = test_model_against_player(device, model)
    print(f"Model wins: {wins}/1000")
