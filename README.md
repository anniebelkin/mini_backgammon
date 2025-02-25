# backgammon

Python modules to play backgammon (human or computer).

## System Requirements

- **Python 3**  
  (You may need to change the commands below to `python3 ...` if that is how you run Python 3 on your machine)

## How to Run the Game

- **Human vs Computer**:  
  Run `python single_player.py`, then choose the computer strategy to play against.
- **Human vs Human**:  
  Run `python two_player.py`.
- **Computer vs Computer**:  
  Run `python main.py` (the two 'players' can have different strategies).
- **Tournament**:  
  Run `python tournament.py`.

The tournament mode runs many games with different starting players and returns the probability of the strategies being equally good.

## Feature Vector and Phase Encoding

Our backgammon project extracts rich features from a board position to feed into our evaluation and reinforcement learning algorithms. The core idea is to represent each board state as a fixed-length numerical vector that captures important aspects of the game and then augment this representation with phase information (early, mid, or late game).

### 16-Dimensional Base Feature Vector

The base feature vector has 16 dimensions and is organized in two halves:

1. **Current Player Features (First 8 Values)**
   - **Occupied Points:**  
     Count of board points (from 1 to 24) where the current player has more than one piece.
   - **Borne-Off Pieces:**  
     Estimated borne-off pieces computed as:  
     `TOTAL_PIECES - (pieces on board) - 2 × (pieces on the bar)`
   - **Pieces on the Bar:**  
     The number of pieces currently waiting to re-enter play.
   - **Blot Distance Sum:**  
     For board points with exactly one piece (a blot), this is the sum of `(BOARD_SIZE - distance)`. This value provides a measure of how vulnerable the blots are.
   - **Total Distance:**  
     The sum of each piece’s distance to home (using each piece’s `spaces_to_home()` value).
   - **Excess Distance:**  
     The sum of distances beyond the home zone. For each piece with `spaces_to_home() > ENDZONE_SIZE`, we add `(spaces_to_home() - ENDZONE_SIZE)`.
   - **Blot Count:**  
     Count of board points with exactly one piece.
   - **Pieces on Board:**  
     Total number of pieces the current player has on the board.

2. **Opponent Features (Last 8 Values)**
   - These are computed similarly for the opponent (obtained via `current_color.other()`):
     - Opponent pieces on board.
     - Opponent blot count.
     - Opponent excess distance.
     - Opponent total distance.
     - Opponent blot distance sum.
     - Opponent pieces on the bar.
     - Opponent borne-off pieces (using the same borne-off formula).
     - Opponent occupied points.

### Phase Calculation and Extended Feature Vector

In addition to the base 16 features, the game phase is determined by analyzing the current player's board state:

- **Pieces Out of Home:**  
  Calculated as  
  `pieces_out_of_home = curr_on_board - curr_pieces_home - curr_on_bar`.
  
- **Average Excess Distance:**  
  Computed as  
  `average_distance = curr_excess_distance / pieces_out_of_home`  
  (if `pieces_out_of_home > 0`).

Based on these values, the phase is assigned as follows:
- **Late Game:**  
  If fewer than 3 pieces are out of home and the average excess distance is less than 12.
- **Mid Game:**  
  If the average excess distance is less than 18 or if the opponent has 3 or more pieces in the home board.
- **Early Game:**  
  Otherwise.

Phase Calculation:
We represent the game phase using an integer:
    0: Early game
    1: Mid game
    2: Late game

To integrate the phase with the base feature vector, the phase integer is converted into a one-hot vector (e.g., early becomes [1, 0, 0]) and is then used to "gate" the base 16-dimensional vector. “gate” the 16 features with the phase vector. That is, we replicate the 16 features three times (one block for each phase) and multiply each block by the corresponding one-hot element. For example, if the game is in the early phase, the final augmented feature vector becomes:
  
[1 × (16 features), 0 × (16 features), 0 × (16 features)]


yielding a 48-dimensional vector. This phase-aware representation allows our neural network to learn a single evaluation function that adapts to different stages of the game.

### Normalization

For training stability, the features can be normalized by dividing by known maximum values:
- **TOTAL_PIECES:** 15 (each side has 15 pieces)
- **BOARD_SIZE:** 25 (used in distance computations)
- **MAX_OCCUPIED_POINTS:** 7 (maximum possible occupied points)
- **MAX_DISTANCES:** 360 (maximum sum of distances)
- **MAX_EXCESS_DISTANCE:** 270 (maximum sum of excess distances)

When normalization is enabled, each feature is scaled to the range [0, 1].

### Implementation Overview

The key functions in `feature_vector.py` are:

- **`board_to_vector(current_color, board, should_normalize)`**  
  Computes the 16-dimensional base feature vector (for both the current player and the opponent) along with the phase (as an integer 0, 1, or 2).

- **`get_phase(...)`**  
  Determines the phase of the current player's board using the logic described above and returns the phase (as an integer 0, 1, or 2).

- **`board_to_extended_vector(player, board)`**  
  Converts the phase integer into a one-hot vector and uses it to produce the final 48-dimensional phase-aware vector.

This design allows our reinforcement learning and evaluation modules to work with a concise, normalized, and phase-aware representation of the board, enabling effective learning and game analysis.

## Our RL Implementation

Our reinforcement learning implementation utilizes the feature vector described above. The feature extraction module (`feature_vector.py`) is responsible for converting the board state into a numerical representation that reflects both static board features and dynamic game phase information. This vector is then used to train our evaluation networks and other RL components.

