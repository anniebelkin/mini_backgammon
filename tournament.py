import random
from random import randint
from src.colour import Colour
from src.game import Game
from src.strategy_factory import StrategyFactory
from src.strategies import HumanStrategy
import sys

def log(message, file_path="tournament_log.txt"):
    with open(file_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)
def print_tournament_placement(player_names):
    log("Tournament Placement:")
    for i in range(0, len(player_names), 2):
        if i + 1 < len(player_names):
            log(f"{player_names[i]} vs {player_names[i + 1]}")
        else:
            log(f"{player_names[i]} gets a bye")

def print_tournament_branch(tournament_branch):
    log("\nTournament Branch:")
    for round_num, matchups in enumerate(tournament_branch, start=1):
        log(f"Round {round_num}:")
        for matchup in matchups:
            log(f"  {matchup}")

def run_tournament(player_names, player_strategies):
    random.shuffle(player_names)
    players = {name: player_strategies[name] for name in player_names} #name : Startegy from strategy dictionary

    print_tournament_placement(player_names)
    
    time_limit = input("Enter time limit in seconds (or 'inf' for no limit): ")
    if time_limit.lower() == 'inf':
        time_limit = -1
    else:
        time_limit = int(time_limit)
    
    while True:
            best_of = input("Enter the number of games for best of series (must be an odd number): ")
            try:
                best_of = int(best_of)
                if best_of % 2 == 1:  # Check if the number is odd
                    break  # Valid input, exit the loop
                else:
                    log("The number of games must be an odd number. Please try again.")
            except ValueError:
                log("Invalid input. Please enter a valid number.")
    
    game_results = []
    tournament_branch = []
    while len(players) > 1:
        next_round_players = {}
        player_names = list(players.keys())
        #random.shuffle(player_names)
        round_matchups = []
        for i in range(0, len(player_names), 2):
            if i + 1 >= len(player_names):
                next_round_players[player_names[i]] = players[player_names[i]]
                continue
            player1 = player_names[i]
            player2 = player_names[i + 1]
            round_matchups.append(f"{player1} vs {player2}")
            log(f"Starting game: {player1} vs {player2}")  
            colour1 = Colour(randint(0, 1))
            colour2 = colour1.other()
            first_player = colour1

            wins = {player1: 0, player2: 0}
            for game_number in range(1, best_of + 1):
                current_first_player = first_player if game_number % 2 != 0 else first_player.other()
                game = Game(
                    white_strategy=players[player1] if colour1 == Colour.WHITE else players[player2],
                    black_strategy=players[player1] if colour1 == Colour.BLACK else players[player2],
                    first_player=current_first_player,
                    time_limit=time_limit
                )
                game.run_game(verbose=True)
                winner = game.who_won()
                winner_name = player1 if winner == colour1 else player2
                wins[winner_name] += 1
                if wins[winner_name] > best_of // 2:
                    break

            series_winner = player1 if wins[player1] > wins[player2] else player2
            next_round_players[series_winner] = players[series_winner]
            game_results.append(f"{player1} vs {player2}: {series_winner} won the series")
            log(f"{series_winner} won the series!")
        tournament_branch.append(round_matchups)
        players = next_round_players
    final_winner = list(players.keys())[0]
    log(f"{final_winner} is the tournament champion!")
    log("\nTournament Results:")
    for result in game_results:
        log(result)
    print_tournament_branch(tournament_branch)

if __name__ == '__main__':
    while True:
        try:
            num_players = int(input("Enter the number of players: "))
            if num_players > 1:
                break
            else:
                log("Tournament must include more than 1 player. Please try again.")
        except ValueError:
            log("Invalid input. Please enter a valid number.")
    log("Available Strategies:")
    strategies = [x for x in StrategyFactory.get_all() if x.__name__ != HumanStrategy.__name__]
    log("[0] HumanStrategy (N/A)")
    for i, strategy in enumerate(strategies):
        log("[%d] %s (%s)" % (i+1, strategy.__name__, strategy.get_difficulty()))
    
    player_names = []
    player_strategies = {}
    for i in range(num_players):
        player_names.append(input('Name of player %d: ' % (i+1)))
        strategy_index = int(input('Pick a strategy for player %d:\n' % (i+1)))
        if strategy_index == 0:
            player_strategies[player_names[i]] = HumanStrategy(player_names[i])
        else:
            chosen_strategy = StrategyFactory.create_by_name(strategies[strategy_index-1].__name__)
            player_strategies[player_names[i]] = chosen_strategy
        

    run_tournament(player_names, player_strategies)
