import torch

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

class BestMoveModel(BestMoveStrategy):
    """A strategy that evaluates board positions using a model."""

    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.model = model
        self.model.eval()

    def evaluate_board(self, board, colour):
        """

        The feature vector is weighted according to the current phase of the game.
        The final score is normalized to the range [0,1].
        """
        feature_vector = board_to_extended_vector(colour, board)
        tensor = torch.tensor(feature_vector, dtype=torch.float32).to(self.device)
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            value = self.model(tensor)
        return value.item()


class RLPlayer_strategy(BestMoveModel):
    def __init__(self, model_path="RL/best_model.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HeuristicNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        super().__init__(model, device)