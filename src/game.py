import json
from random import randint
import threading
from src.board import Board
from src.colour import Colour
from src.strategies import Strategy
from src.move_not_possible_exception import MoveNotPossibleException

def log(message, file_path="tournament_log.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)
class ReadOnlyBoard:
    board: Board

    def __init__(self, board):
        self.board = board

    # Delegate all readonly method calls to the board
    def __getattr__(self, name):
        if hasattr(self.board, name) and callable(getattr(self.board, name)):
            return getattr(self.board, name)

        return super(ReadOnlyBoard, self).__getattr__(name)

    def add_many_pieces(self, number_of_pieces, colour, location):
        self.__raise_exception__()

    def move_piece(self, piece, die_roll):
        self.__raise_exception__()

    def __raise_exception__(self):
        raise Exception("Do not try and change the board directly, use the make_move parameter instead")


class Game:
    def __init__(self, white_strategy: Strategy, black_strategy: Strategy, first_player: Colour, time_limit=5):
        self.board = Board.create_starting_board()
        self.first_player = first_player
        self.strategies = {
            Colour.WHITE: white_strategy,
            Colour.BLACK: black_strategy
        }
        self.time_limit = time_limit
        self.board.time_limit = time_limit

    def run_game(self, verbose=True):
        if verbose:
            log('%s goes first' % self.first_player)
            self.board.print_board()
        i = self.first_player.value
        moves = []
        full_dice_roll = []
        while True:
            if verbose:
                self.board.print_board()
            previous_dice_roll = full_dice_roll.copy()
            dice_roll = [randint(1, 6), randint(1, 6)]
            if dice_roll[0] == dice_roll[1]:
                dice_roll = [dice_roll[0]] * 4
            full_dice_roll = dice_roll.copy()
            colour = Colour(i % 2)
            if verbose:
                log("%s rolled %s" % (colour, dice_roll))

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

            board_snapshot = self.board.to_json()
            dice_roll_snapshot = dice_roll.copy()

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
                if verbose:
                    log('%s did not make a move in time. Skipping turn.' % colour)
                # Skip the turn and continue to the next player
                i += 1
                stop_input_event.clear()  # Clear the stop input event
                continue

            # **Insert the game ending check here**
            if self.board.has_game_ended():  # Check if the game has ended
                if verbose:
                    log(f"{self.board.who_won()} has won the game!")
                stop_input_event.set()  # Stop input in strategies.py as the game is over
                return  # Exit the loop and end the game

            i += 1 #switching between players turns

    def get_rolls_to_move(self, location, requested_move, available_rolls):
        # This first check ensures we return doing as little work as possible when the requested
        # move is exactly one of the die rolls (to ensure automated experiments don't run slower)
        if available_rolls.__contains__(requested_move):
            if self.board.is_move_possible(self.board.get_piece_at(location), requested_move):
                return [requested_move]
            return None
        if len(available_rolls) == 1:
            return None
        
        board = self.board.create_copy()
        rolls_to_move = []
        current_location = location
        if not board.is_move_possible(board.get_piece_at(current_location), available_rolls[0]):
            # If the first die roll isn't possible, reverse the dice before starting. This ensures
            # we cover all possible orderings (becasue doubles will always all be the same number)
            available_rolls = available_rolls.copy()
            available_rolls.reverse()

        for roll in available_rolls:
            piece = board.get_piece_at(current_location)
            if not board.is_move_possible(piece, roll):
                break
            current_location = board.move_piece(piece, roll)
            rolls_to_move.append(roll)
            if sum(rolls_to_move) == requested_move:
                return rolls_to_move
        return None

    def who_started(self):
        return self.first_player

    def who_won(self):
        return self.board.who_won()