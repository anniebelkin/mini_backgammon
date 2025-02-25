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

The base feature vector consists of **16 features**, where the first 8 represent the current player and the last 8 represent the opponent. These include:

- **Occupied Points**: Number of board points (1-24) occupied by two or more pieces.
- **Borne-Off Pieces**: Pieces that have exited the board.
- **Pieces on the Bar**: Pieces waiting to re-enter play.
- **Blot Distance Sum**: Vulnerability of single pieces (blots).
- **Total Distance**: Sum of distances of all pieces to home.
- **Excess Distance**: Extra distance beyond the home zone.
- **Blot Count**: Number of single-piece positions.
- **Pieces on Board**: Total number of pieces in play.

### Phase Calculation and Extended Feature Vector

In addition to the base 16 features, the game phase is determined by analyzing the current player's board state:

- **Pieces Out of Home:**  
  Calculated as  
  `pieces_out = curr_on_board - curr_pieces_home`  
  (i.e., the total pieces on board minus those already in the home area).

Based on this value, the phase is assigned as follows:
- **Late Game:** If fewer than 3 pieces are out of home.
- **Early Game:** If more than 8 pieces are out of home.
- **Mid Game:** Otherwise.

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

## Heuristic Weights and Strategies

### Heuristic Weights

Our heuristic evaluation uses a set of phase-dependent weights that adjust the importance of each feature based on the game phase. The weights are stored in a 16×3 NumPy matrix where each row corresponds to a feature (in the same order as the 16-dimensional feature vector) and each column corresponds to a phase:

| Feature | Early | Mid | Late | Explanation |
|---------|-------|-----|------|-------------|
| **Occupied Points** | 1.3 | 1.7 | 0.8 | Important early to establish strong positions, less valuable late when pieces are moving home. |
| **Borne-Off Pieces** | 3.0 | 5.0 | 8.0 | Increases in importance as the game progresses since winning is determined by bearing off first. |
| **Pieces on Bar** | 0.0 | 0.0 | 0.0 | We do not directly reward or penalize pieces on the bar in this heuristic. |
| **Blot Distance** | -0.5 | -0.3 | -0.1 | Vulnerability decreases in importance as blots become rarer in later phases. |
| **Total Distance** | -0.6 | -0.4 | -0.2 | Distance matters most early when trying to establish a lead, but becomes less relevant when bearing off. |
| **Excess Distance** | -0.7 | -0.9 | -1.3 | More important late game as pieces need to move home faster. |
| **Blot Count** | -0.5 | -0.7 | -0.3 | Avoiding blots is crucial in midgame, where being hit is most costly. |
| **Pieces on Board** | 1.4 | 1.2 | 1.7 | Generally valuable, especially late game when every piece needs to bear off. |
| **Opponent Pieces on Board** | 0.0 | 0.0 | 0.0 | Not considered in this heuristic. |
| **Opponent Blot Count** | 0.0 | 0.0 | 0.0 | Not considered in this heuristic. |
| **Opponent Excess Distance** | 0.0 | 0.0 | 0.0 | Not considered in this heuristic. |
| **Opponent Total Distance** | 0.8 | 1.1 | 0.5 | Important midgame when trying to slow the opponent down. |
| **Opponent Blot Distance** | 0.0 | 0.0 | 0.0 | Not considered in this heuristic. |
| **Opponent Pieces on Bar** | 2.2 | 2.7 | 3.2 | More opponent pieces on the bar is always beneficial. |
| **Opponent Borne-Off Pieces** | 0.0 | 0.0 | 0.0 | Not directly considered in this heuristic. |
| **Opponent Occupied Points** | 0.0 | 0.0 | 0.0 | Not considered in this heuristic. |

When evaluating a board, the phase (returned as an integer between 0 and 2) is used to select the corresponding column of weights. The board evaluation is computed by taking the dot product between the 16-dimensional normalized feature vector and the selected weight vector. The result is then normalized to the range [0, 1] using pre-defined MIN and MAX score norms.

#### Why Min-Max Normalization?

We apply Min-Max normalization to our heuristic scores to ensure consistent scaling and stable training in our reinforcement learning system. The primary reasons are:

1. **Consistency of Scale:**  
   The heuristic produces scores that vary across different board states. Min-Max normalization ensures all scores fall within a fixed range \([0,1]\), making comparisons across positions meaningful.

2. **Preserving Relative Order:**  
   Since our heuristic is based on weighted sums of board features, we need to maintain the order of evaluations—i.e., if one board position is better than another before normalization, it should remain better after normalization. Min-Max scaling preserves these relationships.

3. **Stability in Neural Network Training:**  
   Helps stabilize training by keeping inputs in a well-defined range. Many machine learning models, including reinforcement learning agents, train more effectively when their input values remain between \([0,1]\), preventing extreme values from dominating the learning process.

4. **Handling Negative and Positive Scores:**  
   Our heuristic function produces both positive and negative scores. Min-Max normalization shifts and scales these scores so that all values are mapped to a non-negative range while maintaining their relative differences.

The normalization formula is:  
`score = (score - MIN_SCORE_NORM) / (MAX_SCORE_NORM - MIN_SCORE_NORM)`

Where:
- `MAX_SCORE_NORM` is chosen as the sum of the maximum positive contributions from the heuristic weights.
- `MIN_SCORE_NORM` is chosen as the sum of the maximum negative contributions.

This transformation ensures that the heuristic values remain in a predictable range, making them more suitable for training reinforcement learning models.

### Strategy Classes

#### `BestMoveStrategy`
This is the base class that implements move selection by recursively evaluating all possible moves. It:
- Uses the feature extraction (with phase information) to evaluate board states.
- Explores moves recursively to choose the sequence with the highest evaluation.

#### `BestMoveHeuristic`
This class extends `BestMoveStrategy` by implementing the `evaluate_board` method:
- It obtains the 16-dimensional feature vector and the phase from `board_to_vector()`.
- It selects the appropriate 16-dimensional weight vector from the heuristic weight matrix using the phase.
- It computes the evaluation score as the dot product of the feature vector and the weights.
- Finally, it applies Min-Max normalization to map the score to [0, 1].

## Neural Network for Board Evaluation

We use a fully connected feedforward neural network to evaluate board positions. This network is designed to learn an evaluation function based on a two-stage training process.

### Network Architecture
Our model consists of 3 fully connected layers:

| Layer | Input Size | Output Size | Activation |
|-------|------------|-------------|------------|
| fc1   | 48         | 64          | ReLU       |
| fc2   | 64         | 64          | ReLU       |
| fc3   | 64         | 1           | None (Linear) |

*Note:* We do not use any activation on the final layer. Instead, all target values are pre-normalized to the [0,1] range using Min-Max scaling.

### Training Process

#### Part 1: Heuristic Pre-training
- **Data Collection:**  
  Sample board positions from games played by two random-move players.
- **Target Values:**  
  Use our handcrafted heuristic function (which is normalized to [0,1]) as the target for each board.
- **Objective:**  
  Train the network in a supervised manner so that it learns to approximate our heuristic evaluation.

#### Part 2: Winner/Loser Training
- **Data Collection:**  
  Generate board samples using the heuristic player that selects from the 4 best actions.
- **Target Values:**  
  Label boards based on the eventual outcome: boards from winning positions get high target values, and boards from losing positions get low target values.
- **Objective:**  
  Refine the network by training it to predict outcome-based values, effectively learning to distinguish winning from losing board positions.

### Why This Architecture and Process?
- **3 Layers (with two hidden layers of size 64):**  
  Provides enough capacity to capture complex board features and strategic nuances.
- **No Activation on the Final Layer:**  
  Allows the model to produce a raw scalar output, while target values are normalized to [0,1] during training.
- **Two-Stage Training:**  
  - **Stage 1:** Provides a strong starting point by mimicking the handcrafted heuristic.
  - **Stage 2:** Improves performance by focusing on actual game outcomes (winner vs. loser) using a more selective sample of board positions.

This approach ensures the network first learns a good approximation of our heuristic and then refines its evaluation to better predict winning outcomes.

