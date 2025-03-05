import torch
import torch.nn as nn
from src.strategies import Strategy
from src.piece import Piece
from RL.feature_vector import board_to_vector, board_to_extended_vector

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

class BestMoveStrategy(Strategy):
    """Backgammon strategy that selects the best move using a heuristic evaluation."""

    @staticmethod
    def get_difficulty():
        """Return the difficulty level of the strategy."""
        return "Hard"

    def move(self, board, colour, dice_roll, make_move, opponents_activity):
        """
        Determine and execute the best move sequence based on heuristic evaluation.
        
        If rolling two dice, it tries both orders to find the optimal move.
        """
        result = self.move_recursively(board, colour, dice_roll)
        if len(dice_roll) == 2:
            new_dice_roll = dice_roll[::-1]
            result_swapped = self.move_recursively(board, colour, new_dice_roll)
            if result_swapped['best_value'] > result['best_value'] and \
                    len(result_swapped['best_moves']) >= len(result['best_moves']):
                result = result_swapped

        for move in result['best_moves']:
            make_move(move['piece_at'], move['die_roll'])

    def move_recursively(self, board, colour, dice_rolls):
        """
        Recursively explore all possible move sequences to find the best board evaluation.
        
        Returns a dictionary with:
        - 'best_value': the highest board evaluation score found.
        - 'best_moves': the sequence of moves leading to the best evaluation.
        """
        best_board_value = float('-inf')
        best_pieces_to_move = []

        # Get all unique piece locations
        pieces_to_try = list(set(piece.location for piece in board.get_pieces(colour)))

        # Sort pieces by their distance to home (farther pieces first)
        valid_pieces = [board.get_piece_at(loc) for loc in pieces_to_try]
        valid_pieces.sort(key=Piece.spaces_to_home, reverse=True)

        dice_rolls_left = dice_rolls.copy()
        die_roll = dice_rolls_left.pop(0)

        for piece in valid_pieces:
            if board.is_move_possible(piece, die_roll):
                board_copy = board.create_copy()
                new_piece = board_copy.get_piece_at(piece.location)
                board_copy.move_piece(new_piece, die_roll)

                if dice_rolls_left:
                    result = self.move_recursively(board_copy, colour, dice_rolls_left)
                    if not result['best_moves']:
                        board_value = self.evaluate_board(board_copy, colour)
                        if board_value > best_board_value and len(best_pieces_to_move) < 2:
                            best_board_value = board_value
                            best_pieces_to_move = [{'die_roll': die_roll, 'piece_at': piece.location}]
                    elif result['best_value'] > best_board_value:
                        new_best_moves_length = len(result['best_moves']) + 1
                        if new_best_moves_length >= len(best_pieces_to_move):
                            best_board_value = result['best_value']
                            move = {'die_roll': die_roll, 'piece_at': piece.location}
                            best_pieces_to_move = [move] + result['best_moves']
                else:
                    board_value = self.evaluate_board(board_copy, colour)
                    if board_value > best_board_value and len(best_pieces_to_move) < 2:
                        best_board_value = board_value
                        best_pieces_to_move = [{'die_roll': die_roll, 'piece_at': piece.location}]

        return {'best_value': best_board_value, 'best_moves': best_pieces_to_move}

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