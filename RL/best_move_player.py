import numpy as np

from src.strategies import Strategy
from src.piece import Piece
from RL.feature_vector import board_to_vector

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
