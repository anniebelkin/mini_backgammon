import torch
from RL.best_move_player import BestMoveModel
from RL.train import HeuristicNN

class RLPlayer_strategy(BestMoveModel):
    def __init__(self, model_path="RL/best_model.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HeuristicNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        super().__init__(model, device)