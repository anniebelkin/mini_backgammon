from random import shuffle
import copy
import json
import threading
import tkinter as tk
from PIL import Image, ImageTk

from src.colour import Colour
from src.piece import Piece

def log(message, file_path="tournament_log.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)
class Board:
    def __init__(self):
        self.__pieces = []
        self.time_winner = None  
        self.gui_initialized = False
        self.time_limit = -1

    @classmethod
    def create_starting_board(cls):
        board = Board()
        board.add_many_pieces(2, Colour.WHITE, 1)
        board.add_many_pieces(5, Colour.BLACK, 6)
        board.add_many_pieces(3, Colour.BLACK, 8)
        board.add_many_pieces(5, Colour.WHITE, 12)
        board.add_many_pieces(5, Colour.BLACK, 13)
        board.add_many_pieces(3, Colour.WHITE, 17)
        board.add_many_pieces(5, Colour.WHITE, 19)
        board.add_many_pieces(2, Colour.BLACK, 24)
        return board

    def add_many_pieces(self, number_of_pieces, colour, location):
        for _ in range(number_of_pieces):
            self.__pieces.append(Piece(colour, location))

    def is_move_possible(self, piece, die_roll):
        if len(self.pieces_at(self.__taken_location(piece.colour))) > 0:
            if piece.location != self.__taken_location(piece.colour):
                return False
        if piece.colour == Colour.BLACK:
            die_roll = -die_roll
        new_location = piece.location + die_roll
        if new_location <= 0 or new_location >= 25:
            if not self.can_move_off(piece.colour):
                return False
            if new_location != 0 and new_location != 25:
                # this piece will overshoot the end
                return not any(x.spaces_to_home() >= abs(die_roll) for x in self.get_pieces(piece.colour))
            return True
        pieces_at_new_location = self.pieces_at(new_location)
        if len(pieces_at_new_location) == 0 or len(pieces_at_new_location) == 1:
            return True
        return self.pieces_at(new_location)[0].colour == piece.colour

    def no_moves_possible(self, colour, dice_roll):
        piece_locations = [x.location for x in self.get_pieces(colour)]
        piece_locations = list(set(piece_locations))

        dice_roll = list(set(dice_roll))

        pieces = []
        for piece_location in piece_locations:
            pieces.append(self.get_piece_at(piece_location))
        for die in dice_roll:
            for piece in pieces:
                if self.is_move_possible(piece, die):
                    return False

        return True

    def can_move_off(self, colour):
        return all(x.spaces_to_home() <= 6 for x in self.get_pieces(colour))

    def move_piece(self, piece, die_roll):
        if not self.__pieces.__contains__(piece):
            raise Exception('This piece does not belong to this board')
        if not self.is_move_possible(piece, die_roll):
            raise Exception('You cannot make this move')
        if piece.colour == Colour.BLACK:
            die_roll = -die_roll

        new_location = piece.location + die_roll
        if new_location <= 0 or new_location >= 25:
            self.__remove_piece(piece)

        pieces_at_new_location = self.pieces_at(new_location)

        if len(pieces_at_new_location) == 1 and pieces_at_new_location[0].colour != piece.colour:
            piece_to_take = pieces_at_new_location[0]
            piece_to_take.location = self.__taken_location(piece_to_take.colour)

        piece.location = new_location
        return new_location

    def pieces_at(self, location):
        return [x for x in self.__pieces if x.location == location]

    def get_piece_at(self, location):
        pieces = self.pieces_at(location)
        if len(pieces) == 0:
            return None
        return pieces[0]

    def get_pieces(self, colour):
        pieces = [x for x in self.__pieces if x.colour == colour]
        shuffle(pieces)
        return pieces

    def get_taken_pieces(self, colour):
        return self.pieces_at(self.__taken_location(colour))

    def has_game_ended(self):
        return self.time_winner is not None or len(self.get_pieces(Colour.WHITE)) == 0 or len(self.get_pieces(Colour.BLACK)) == 0

    def who_won(self):
        if not self.has_game_ended():
            raise Exception('The game has not finished yet!')
        if self.time_winner is not None:
            return self.time_winner
        return Colour.WHITE if len(self.get_pieces(Colour.WHITE)) == 0 else Colour.BLACK

    def create_copy(self):
        return copy.deepcopy(self)

    def get_move_lambda(self):
        return lambda l, r: self.move_piece(self.get_piece_at(l), r)

    def initialize_gui(self):
        self.root = tk.Tk()
        self.root.title("Backgammon Board")
        self.canvas = tk.Canvas(self.root, width=1280, height=720)
        self.canvas.pack()
        self.board_img = Image.open("./pic/board.PNG").resize((1280, 720), Image.Resampling.LANCZOS)
        self.board_img = ImageTk.PhotoImage(self.board_img)
        self.white_img = Image.open("./pic/white.PNG").resize((65, 65), Image.Resampling.LANCZOS)
        self.white_img = ImageTk.PhotoImage(self.white_img)
        self.black_img = Image.open("./pic/black.PNG").resize((65, 65), Image.Resampling.LANCZOS)
        self.black_img = ImageTk.PhotoImage(self.black_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.board_img)
        self.gui_initialized = True

    def update_gui(self):
        if not self.gui_initialized:
            self.initialize_gui()
        self.canvas.delete("piece")
        for location in range(1, 25):
            pieces = self.pieces_at(location)
            for i, piece in enumerate(pieces):
                img = self.white_img if piece.colour == Colour.WHITE else self.black_img
                x, y = self.get_gui_coordinates(location, i)
                self.canvas.create_image(x, y, anchor=tk.NW, image=img, tags="piece")
        self.root.update()

    def get_gui_coordinates(self, location, index=0):
        if location == 0:
            return (50, 350 + index * 70)
        elif location == 25:
            return (750, 350 + index * 70)
        else:
            if location >= 13:
                x = 15 + (location - 13) * 100
                y = 0 + index * 65
            else:
                x = 15 + (12 - location) * 100
                y = 655 - index * 65
            return (x, y)

    def start_gui_thread(self):
        threading.Thread(target=self.run_gui, daemon=True).start()

    def run_gui(self):
        self.root.mainloop()

    def print_board(self):
        log("  13                  18   19                  24   25")
        log("---------------------------------------------------")
        line = "|"
        for i in range(13, 18 + 1):
            line = line + self.__pieces_at_text(i)
        line = line + "|"
        for i in range(19, 24 + 1):
            line = line + self.__pieces_at_text(i)
        line = line + "|"
        line = line + self.__pieces_at_text(self.__taken_location(Colour.BLACK))
        log(line)
        for _ in range(3):
            log("|                        |                        |")
        line = "|"
        for i in reversed(range(7, 12+1)):
            line = line + self.__pieces_at_text(i)
        line = line + "|"
        for i in reversed(range(1, 6+1)):
            line = line + self.__pieces_at_text(i)
        line = line + "|"
        line = line + self.__pieces_at_text(self.__taken_location(Colour.WHITE))
        log(line)
        log("---------------------------------------------------")
        log("  12                  7    6                   1    0")
        #self.update_gui()

    def to_json(self):
        data = {}
        for location in range(26):
            pieces = self.pieces_at(location)
            if len(pieces) > 0:
                data[location] = {'colour': pieces[0].colour.__str__(), 'count': len(pieces)}
        return json.dumps(data)

    def export_state(self):
        state = [0] * 28
        for location in range(1, 25):
            pieces = self.pieces_at(location)
            if len(pieces) > 0:
                state[location - 1] = len(pieces) if pieces[0].colour == Colour.WHITE else -len(pieces)
        state[24] = len(self.pieces_at(0))  # White pieces eaten
        state[25] = len(self.pieces_at(25))  # Black pieces eaten
        state[26] = 15-len(self.get_pieces(Colour.WHITE))  
        state[27] = 15-len(self.get_pieces(Colour.BLACK))  
        return state

    def import_state(self, state):
        self.__pieces = []
        for location, count in enumerate(state[:24]):
            if count > 0:
                self.add_many_pieces(count, Colour.WHITE, location + 1)
            elif count < 0:
                self.add_many_pieces(-count, Colour.BLACK, location + 1)
        self.add_many_pieces(state[24], Colour.WHITE, 0)  # White pieces blown
        self.add_many_pieces(state[25], Colour.BLACK, 25)  # Black pieces blown
       

    def __taken_location(self, colour):
        if colour == Colour.WHITE:
            return 0
        else:
            return 25

    def __pieces_at_text(self, location):
        pieces = self.pieces_at(location)
        if len(pieces) == 0:
            return " .  "
        if pieces[0].colour == Colour.WHITE:
            return " %sW " % (len(pieces))
        else:
            return " %sB " % (len(pieces))

    def __remove_piece(self, piece):
        self.__pieces.remove(piece)
    
    def getTheTimeLim(self):
        return self.time_limit
