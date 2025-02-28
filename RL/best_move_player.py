import numpy as np
import math
import random

import torch

from src.strategies import Strategy
from src.piece import Piece
from RL.feature_vector import board_to_vector, board_to_extended_vector

# Heuristic weight matrix for [early, mid, late] game phases
HEURISTIC_WEIGHTS = np.array([
    [1.3, 1.7, 0.8],    # curr_occupied_points
    [3.0, 5.0, 8.0],    # curr_borne_off
    [0.0, 0.0, 0.0],    # curr_on_bar
    [-0.5, -0.3, -0.1], # curr_blot_distance
    [-0.6, -0.4, -0.2], # curr_total_distance
    [-0.7, -0.9, -1.3], # curr_excess_distance
    [-0.5, -0.7, -0.3], # curr_blot_count
    [1.4, 1.2, 1.7],    # curr_on_board
    [0.0, 0.0, 0.0],    # opp_on_board
    [0.0, 0.0, 0.0],    # opp_blot_count
    [0.0, 0.0, 0.0],    # opp_excess_distance
    [0.8, 1.1, 0.5],    # opp_total_distance
    [0.0, 0.0, 0.0],    # opp_blot_distance
    [2.2, 2.7, 3.2],    # opp_on_bar
    [0.0, 0.0, 0.0],    # opp_borne_off
    [0.0, 0.0, 0.0]     # opp_occupied_points
])

# Normalization values for heuristic evaluation
MAX_SCORE_NORM = 14.5
MIN_SCORE_NORM = -2.5

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
    

class BestMoveHeuristic(BestMoveStrategy):
    """A strategy that evaluates board positions using a weighted heuristic function."""

    def evaluate_board(self, board, colour):
        """
        Compute the heuristic evaluation score for a given board state.
        
        The feature vector is weighted according to the current phase of the game.
        The final score is normalized to the range [0,1].
        """
        feature_vector, phase = board_to_vector(colour, board, should_normalize=True)
        weights_vector = HEURISTIC_WEIGHTS[:, phase]
        score = np.dot(feature_vector, weights_vector)

        # Min-Max normalization to [0,1] range
        score = (score - MIN_SCORE_NORM) / (MAX_SCORE_NORM - MIN_SCORE_NORM)
        return score
    
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
    
class BestOfFourStrategy(Strategy):
    """
    A strategy that uses a neural network to evaluate board positions.
    It recursively generates move sequences, selects the top 4 moves based
    on the network's evaluation, and then chooses randomly among them using
    exponential weighting.
    """
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.evaluate_board = self._evaluate_board

    def _evaluate_board(self, board, colour):
        feature_vector = board_to_extended_vector(colour, board)
        tensor = torch.tensor(feature_vector, dtype=torch.float32).to(self.device)
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            value = self.model(tensor)
        return value.item()

    @staticmethod
    def get_difficulty():
        return "Hard"

    def move(self, board, colour, dice_roll, make_move, opponents_activity):
        pieces = board.get_pieces(colour)
        # Special case: if only one piece remains.
        if len(pieces) == 1:
            piece = pieces[0]
            best_value = float('-inf')
            best_die = None
            for d in dice_roll:
                if board.is_move_possible(piece, d):
                    board_copy = board.create_copy()
                    new_piece = board_copy.get_piece_at(piece.location)
                    board_copy.move_piece(new_piece, d)
                    board_value = self.evaluate_board(board_copy, colour)
                    if board_value > best_value:
                        best_value = board_value
                        best_die = d
            if best_die is not None:
                make_move(piece.location, best_die)
            return

        # Generate move sequences.
        combos1 = self.move_recursively(board, colour, dice_roll)
        if len(dice_roll) == 2:
            new_dice_roll = dice_roll.copy()
            new_dice_roll.reverse()
            combos2 = self.move_recursively(board, colour, new_dice_roll)
            combined = combos1 + combos2
            combined.sort(key=lambda combo: combo['value'], reverse=True)
            top_combos = combined[:4]
        else:
            top_combos = combos1

        if top_combos:
            if len(top_combos) < 4:
                chosen_combo = top_combos[0]
            else:
                weights = [math.exp(combo['value']) for combo in top_combos]
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                chosen_combo = random.choices(top_combos, weights=normalized_weights, k=1)[0]
            for move_info in chosen_combo['moves']:
                make_move(move_info['piece_at'], move_info['die_roll'])

    def move_recursively(self, board, colour, dice_rolls):
        if not dice_rolls:
            board_value = self.evaluate_board(board, colour)
            return [{'value': board_value, 'moves': []}]

        dice_rolls_left = dice_rolls.copy()
        die_roll = dice_rolls_left.pop(0)
        move_combinations = []
        pieces_to_try = list({piece.location for piece in board.get_pieces(colour)})
        valid_pieces = [board.get_piece_at(loc) for loc in pieces_to_try]
        valid_pieces.sort(key=Piece.spaces_to_home, reverse=True)

        for piece in valid_pieces:
            if board.is_move_possible(piece, die_roll):
                board_copy = board.create_copy()
                new_piece = board_copy.get_piece_at(piece.location)
                board_copy.move_piece(new_piece, die_roll)
                if dice_rolls_left:
                    subsequent_combos = self.move_recursively(board_copy, colour, dice_rolls_left)
                    for combo in subsequent_combos:
                        new_moves = [{'die_roll': die_roll, 'piece_at': piece.location}] + combo['moves']
                        move_combinations.append({'value': combo['value'], 'moves': new_moves})
                else:
                    board_value = self.evaluate_board(board_copy, colour)
                    new_moves = [{'die_roll': die_roll, 'piece_at': piece.location}]
                    move_combinations.append({'value': board_value, 'moves': new_moves})
        move_combinations.sort(key=lambda combo: combo['value'], reverse=True)
        return move_combinations[:4]
