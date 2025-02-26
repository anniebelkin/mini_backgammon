import json
import threading
import random
from random import randint
from tqdm import tqdm
from src.game import Game, ReadOnlyBoard
from src.colour import Colour
from src.move_not_possible_exception import MoveNotPossibleException
from RL.feature_vector import board_to_extended_vector as get_board_vector
from src.strategies import MoveRandomPiece
from RL.best_move_player import BestMoveHeuristic, BestOfFourStrategy, BestMoveModel


def log(message, file_path="tournament_log.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

class RecordGame(Game):
    def run_game(self, sample_file_name = "RL/board_samples"):
        heuristic_function = BestMoveHeuristic()

        i = self.first_player.value
        moves = []
        full_dice_roll = []
        training_data = []
        while True:
            previous_dice_roll = full_dice_roll.copy()
            dice_roll = [randint(1, 6), randint(1, 6)]
            if dice_roll[0] == dice_roll[1]:
                dice_roll = [dice_roll[0]] * 4
            full_dice_roll = dice_roll.copy()
            colour = Colour(i % 2)

            # This nested function executes a move (which can consist of one or more dice rolls)
            def handle_move(location, die_roll):
                rolls_to_move = self.get_rolls_to_move(location, die_roll, dice_roll)
                if rolls_to_move is None:
                    raise MoveNotPossibleException("You cannot move that piece %d" % die_roll)
                for roll in rolls_to_move:
                    piece = self.board.get_piece_at(location)
                    original_location = location
                    location = self.board.move_piece(piece, roll)
                    dice_roll.remove(roll)
                    moves.append({'start_location': original_location, 'die_roll': roll, 'end_location': location})
                    previous_dice_roll.append(roll)
                return rolls_to_move

            opponents_moves = moves.copy()
            moves.clear()

            move_made = threading.Event()
            # Pass the stop event to the strategy
            stop_input_event = threading.Event()
            self.strategies[colour].stop_input_event = stop_input_event

            def make_move():
                self.strategies[colour].move(
                    ReadOnlyBoard(self.board),
                    colour,
                    dice_roll.copy(),
                    lambda location, die_roll: handle_move(location, die_roll),
                    {'dice_roll': previous_dice_roll, 'opponents_move': opponents_moves}
                )
                move_made.set()

            move_thread = threading.Thread(target=make_move)
            move_thread.start()
            move_thread.join(self.time_limit if self.time_limit > 0 else None)

            if self.time_limit > 0 and not move_made.is_set():
                stop_input_event.set()  # Stop input in strategies.py
                # Skip the turn and continue to the next player.
                i += 1
                stop_input_event.clear()  # Clear the stop input event.
                continue

            # After the player's turn is completed, record the current board vector.
            vector = get_board_vector(colour, self.board)
            # Save the example with a default target value of 0.
            training_data.append({
                "board_vector": vector,
                "heuristic": heuristic_function.evaluate_board(self.board, colour),
                "target": 0, 
                # Save the colour (as a string) so we can update later.
                "colour": "white" if colour == Colour.WHITE else "black"
            })

            # Check if the game has ended.
            if self.board.has_game_ended():
                winning_colour = self.board.who_won()
                stop_input_event.set()  # Stop input in strategies.py as the game is over.

                # Update all training examples: if the record is for the winning side, set the computed target.
                for record in training_data:
                    if (record["colour"] == "white" and winning_colour == Colour.WHITE) or \
                    (record["colour"] == "black" and winning_colour == Colour.BLACK):
                        record["target"] = 1

                # Append all training examples to the file 'training_boards'.
                # Each line will be a JSON object.
                with open(sample_file_name, "a") as f:
                    for record in training_data:
                        f.write(json.dumps(record) + "\n")
                return  # End the game.

            i += 1  # Switch to the other player's turn.

def sample(first_player, second_player, games, sample_file_name):
    print(f"\nSimulating {games} training games to generate new board samples...\n")

    # Randomly assign strategies.
    if random.choice([True, False]):
        white_strategy = first_player()
        black_strategy = second_player()
    else:
        white_strategy = second_player()
        black_strategy = first_player()

    for _ in tqdm(range(games)):
        
        # Use RecordGame to record boards with our updated target computation.
        game = RecordGame(
            white_strategy=white_strategy,  # or white_strategy
            black_strategy=black_strategy,  # or black_strategy
            first_player= Colour(random.randint(0, 1)),
            time_limit=5,
        )
        game.run_game(sample_file_name = sample_file_name)

def sample_random(games = 200, sample_file_name = "RL/board_random_samples"):
    sample(MoveRandomPiece, MoveRandomPiece, games, sample_file_name)

def sample_best_of_four(games = 200, sample_file_name = "RL/board_samples"):
    sample(BestOfFourStrategy, BestOfFourStrategy, games, sample_file_name)

def test_model(device, model, opponent_player, opponent_model=None):
    wins = 0
    # Randomly assign strategies.
    if random.choice([True, False]):
        white_strategy = opponent_player(opponent_model, device) if opponent_model else opponent_player()
        black_strategy = BestMoveModel(model, device)
        color = Colour.BLACK
    else:
        white_strategy = BestMoveModel(model, device)
        black_strategy = opponent_player(opponent_model, device) if opponent_model else opponent_player()
        color = Colour.WHITE

    for _ in tqdm(range(10)):
        game = Game(
            white_strategy=white_strategy,
            black_strategy=black_strategy,
            first_player= Colour(random.randint(0, 1)),
            time_limit=5
        )
        game.run_game(verbose=False)
        if game.who_won() == color:
            wins += 1
    avg_wins = wins/10
    print(f"Model wins: {wins}/10")
    return avg_wins

def test_model_improvement(device, model, last_saved_model=None):
    print(f"\nTesting the model against last saved model...\n")
    return test_model(device, model, BestMoveModel, last_saved_model)

def test_model_against_heuristic(device, model):
    print(f"\nTesting the model against Heuristic player...\n")
    return test_model(device, model, BestMoveHeuristic)