TOTAL_PIECES = 15
BOARD_SIZE = 25          # Points 1..24
ENDZONE_SIZE = 6         # Pieces with spaces_to_home() <= ENDZONE_SIZE are "in home"
MAX_OCCUPIED_POINTS = 7
MAX_DISTANCES = 360
MAX_EXCESS_DISTANCE = 270

EARLY_PHASE = 0
MID_PHASE = 1
LATE_PHASE = 2

def board_to_vector(current_color, board, should_normalize=False):
    """Return a tuple (board_vector, phase_vector) for the given board."""
    # Current player accumulators
    curr_on_board = 0; curr_total_distance = 0; curr_excess_distance = 0
    curr_blot_count = 0; curr_blot_distance = 0; curr_occupied_points = 0; curr_pieces_home = 0
    # Opponent accumulators
    opp_on_board = 0; opp_total_distance = 0; opp_excess_distance = 0
    opp_blot_count = 0; opp_blot_distance = 0; opp_occupied_points = 0; opp_pieces_home = 0

    opponent_color = current_color.other()
    
    for point in range(1, BOARD_SIZE):  # Points 1..24
        pieces = board.pieces_at(point)
        if not pieces:
            continue
        count = len(pieces)
        d = pieces[0].spaces_to_home()
        piece_color = pieces[0].colour
        
        if piece_color == current_color:
            curr_on_board += count
            curr_total_distance += d * count
            if d > ENDZONE_SIZE:
                curr_excess_distance += (d - ENDZONE_SIZE) * count
            else:
                curr_pieces_home += count
            if count == 1:
                curr_blot_count += 1
                curr_blot_distance += (BOARD_SIZE - d)
            elif count > 1:
                curr_occupied_points += 1
        elif piece_color == opponent_color:
            opp_on_board += count
            opp_total_distance += d * count
            if d > ENDZONE_SIZE:
                opp_excess_distance += (d - ENDZONE_SIZE) * count
            else:
                opp_pieces_home += count
            if count == 1:
                opp_blot_count += 1
                opp_blot_distance += (BOARD_SIZE - d)
            elif count > 1:
                opp_occupied_points += 1

    curr_on_bar = len(board.get_taken_pieces(current_color))
    opp_on_bar = len(board.get_taken_pieces(opponent_color))
    
    curr_borne_off = TOTAL_PIECES - curr_on_board - 2 * curr_on_bar
    opp_borne_off = TOTAL_PIECES - opp_on_board - 2 * opp_on_bar

    norm_total = TOTAL_PIECES if should_normalize else 1
    norm_occupied = MAX_OCCUPIED_POINTS if should_normalize else 1
    norm_total_dist = MAX_DISTANCES if should_normalize else 1
    norm_excess = MAX_EXCESS_DISTANCE if should_normalize else 1

    board_vector = [
        curr_occupied_points / norm_occupied,
        curr_borne_off / norm_total,
        curr_on_bar / norm_total,
        curr_blot_distance / norm_excess,
        curr_total_distance / norm_total_dist,
        curr_excess_distance / norm_excess,
        curr_blot_count / norm_total,
        curr_on_board / norm_total,
        opp_on_board / norm_total,
        opp_blot_count / norm_total,
        opp_excess_distance / norm_excess,
        opp_total_distance / norm_total_dist,
        opp_blot_distance / norm_excess,
        opp_on_bar / norm_total,
        opp_borne_off / norm_total,
        opp_occupied_points / norm_occupied
    ]
    
    phase = get_phase(curr_on_board, curr_pieces_home)
    return board_vector, phase

def get_phase(curr_on_board, curr_pieces_home):
    """
    Determine game phase based on how many pieces are still out of home.
    
    - Late Phase: Few pieces out (e.g., < 3)
    - Early Phase: Most pieces are still out (e.g., > 8)
    - Mid Phase: Otherwise
    """
    pieces_out = curr_on_board - curr_pieces_home
    if pieces_out < 3:
        return LATE_PHASE
    elif pieces_out > 8:
        return EARLY_PHASE
    else:
        return MID_PHASE

def board_to_extended_vector(player, board):
    """Return a 48-dimensional extended vector by gating the 16-dimensional board vector with the phase."""
    board_vector, phase = board_to_vector(player, board, should_normalize=True)
    
    # Convert phase integer to one-hot vector
    phase_vector = [0, 0, 0]
    phase_vector[phase] = 1
    
    extended = []
    for p in phase_vector:
        extended.extend([p * f for f in board_vector])
    return extended
