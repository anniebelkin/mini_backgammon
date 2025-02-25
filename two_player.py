from random import randint

from src.colour import Colour
from src.game import Game
from src.strategies import HumanStrategy


if __name__ == '__main__':
    players = {
        Colour.WHITE: input('Name of player 1: '),
        Colour.BLACK: input('Name of player 2: '),
    }

    time_limit = input("Enter time limit in seconds (or 'inf' for no limit): ")
    if time_limit.lower() == 'inf':
        time_limit = -1
    else:
        time_limit = int(time_limit)

    game = Game(
        white_strategy=HumanStrategy(players[Colour.WHITE]),
        black_strategy=HumanStrategy(players[Colour.BLACK]),
        first_player=Colour(randint(0, 1)),
        time_limit=time_limit
    )

    game.run_game(verbose=False)

    print("%s won!" % players[game.who_won()])
    game.board.print_board()
