import os
import sys
import pickle
import gzip
import csv
import datetime


ROWS = 6
COLS = 7

# ANSI color codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

RED     = '\033[91m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
BLUE    = '\033[94m'
MAGENTA = '\033[95m'
CYAN    = '\033[96m'
WHITE   = '\033[97m'


USE_COLORS = True  # set to False if your terminal is weird


def create_board():
    """Return an empty ROWS x COLS board filled with 0s."""
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]

def check_win(board, player):
    """Return True if `player` has 4 in a row somewhere."""
    # Horizontal check
    for r in range(ROWS):
        for c in range(COLS - 3):
            if (board[r][c] == player and
                board[r][c+1] == player and
                board[r][c+2] == player and
                board[r][c+3] == player):
                return True

    # Vertical check
    for c in range(COLS):
        for r in range(ROWS - 3):
            if (board[r][c] == player and
                board[r+1][c] == player and
                board[r+2][c] == player and
                board[r+3][c] == player):
                return True

    # Diagonal (down-right)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if (board[r][c] == player and
                board[r+1][c+1] == player and
                board[r+2][c+2] == player and
                board[r+3][c+3] == player):
                return True

    # Diagonal (up-right)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if (board[r][c] == player and
                board[r-1][c+1] == player and
                board[r-2][c+2] == player and
                board[r-3][c+3] == player):
                return True

    return False

def is_draw(board):
    """Return True if the board is full and no one has won."""
    return len(get_valid_moves(board)) == 0

def get_valid_moves(board):
    """Return a list of column indices (0-based) that are not full."""
    return [c for c in range(COLS) if board[0][c] == 0]

def drop_piece(board, col, player):
    """
    Drop a piece for `player` (1 or 2) into column `col` (0-based).
    Returns True if successful, False if column is full.
    """
    for r in range(ROWS-1, -1, -1):  # start from bottom row
        if board[r][col] == 0:
            board[r][col] = player
            return True
    return False  # column full

def print_board(board, current_player=None):
    clear_screen()

    # Title
    title = " CONNECT FOUR "
    print()
    print(color("╔" + "═" * len(title) + "╗", CYAN, BOLD))
    print(color("║", CYAN, BOLD) + color(title, BOLD, WHITE) + color("║", CYAN, BOLD))
    print(color("╚" + "═" * len(title) + "╝", CYAN, BOLD))
    print()

    # Turn indicator
    if current_player is not None:
        symbol = "X" if current_player == 1 else "O"
        piece = render_cell(current_player)
        print(f" Turn: Player {current_player} ({symbol}) {piece}\n")

    # Column numbers
    print("  " + "  " + " ".join(str(c + 1) for c in range(COLS)))

    inner_width = 2 * COLS + 1
    print("  " + color("┌" + ("─" * inner_width) + "┐", CYAN))

    # Rows
    for r in range(ROWS):
        row_pieces = [render_cell(board[r][c]) for c in range(COLS)]
        line = (
            "  "
            + color("│", CYAN)
            + " "
            + " ".join(row_pieces)
            + " "
            + color("│", CYAN)
        )
        print(line)

    # Bottom border
    print("  " + color("└" + ("─" * inner_width) + "┘", CYAN))
    print()

def render_cell(value):
    """Return a pretty colored string for a single cell (fixed-width)."""
    if value == 0:
        return color(".", DIM, WHITE)        # empty
    elif value == 1:
        return color("X", BOLD, RED)         # Player 1
    else:
        return color("O", BOLD, YELLOW)      # Player 2

def clear_screen():
    # Try to clear the terminal in a cross-platform way
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    

def color(text, *codes):
    if not USE_COLORS:
        return text
    return "".join(codes) + text + RESET

def board_to_vector(board, player):
    vector = []
    for row in board:
        for col in row:
            if col == player:
                vector.append(1)
            else:
                vector.append(0)
    for row in board:
        for col in row:
            if col != player and col != 0:
                vector.append(1)
            else:
                vector.append(0)
    return vector

def predict_move(board, player, model, use_mcts: bool = True, simulations: int = 200):
    """
    Pick a move for `player` using either MCTS+NN (default) or greedy 1-ply NN.
    """
    if use_mcts:
        from mcts import mcts_move
        return mcts_move(board, player, model, num_simulations=simulations)
    
    # Fallback: simple greedy NN
    moves = get_valid_moves(board)
    best_score = -1
    best_col = None

    for col in moves:
        # make a COPY of the board before simulating
        new_board = [row[:] for row in board]
        drop_piece(new_board, col, player)

        x = board_to_vector(new_board, player)
        score = model.forward(x)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _play_match(model1, model2, games, e1=0.0, e2=0.0):
    from self_play import self_play
    """
    Play `games` games of model1 vs model2 using self_play and
    return (wins1, wins2, draws).

    model1 is ALWAYS Player 1.
    model2 is ALWAYS Player 2.
    Boards are NOT displayed; states are NOT recorded.
    """
    win1 = win2 = draw = 0
    for _ in range(games):
        # we don't care about states here, only the result
        result, _states = self_play(
            model1,
            model2,
            e1=e1,
            e2=e2,
            record_states=False,
            display=False,   # <- important: no printing during eval
        )
        if result == 1:
            win1 += 1  # Player 1 (model1) wins
        elif result == 2:
            win2 += 1  # Player 2 (model2) wins
        else:
            draw += 1
    return win1, win2, draw

def head_to_head(model_new,
                 model_best,
                 games=40,
                 simulations=50,
                 label_new="model_new",
                 label_best="model_best",
                 tag=None,
                 csv_path="csv/head_to_head_log.csv",
                 use_dirichlet=False, dirichlet_alpha=0.3, dirichlet_eps=0.25):
    """
    Deterministic head-to-head match between two models using self_play.

    - model_new vs model_best
    - Alternates colors each game to balance first-player advantage.
    - Uses e1 = e2 = 0.0 (no epsilon randomness).
    - Uses MCTS with `simulations` simulations.
    - Logs results to a CSV file in the csv/ folder.
    - Prints a colorful diagnostic summary.

    Returns (wins_new, wins_best, draws).
    """
    from self_play import self_play

    if games <= 0:
        return 0, 0, 0

    wins_new = 0
    wins_best = 0
    draws = 0

    # Run games, alternating who is Player 1
    for g in range(games):
        if g % 2 == 0:
            # model_new as Player 1, model_best as Player 2
            result, _ = self_play(
                model_new, model_best,
                e1=0.0, e2=0.0,
                record_states=False,
                display=False,
                simulations=simulations,
                use_dirichlet=use_dirichlet,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps
            )
            if result == 1:
                wins_new += 1
            elif result == 2:
                wins_best += 1
            else:
                draws += 1
        else:
            # model_best as Player 1, model_new as Player 2
            result, _ = self_play(
                model_best, model_new,
                e1=0.0, e2=0.0,
                record_states=False,
                display=False,
                simulations=simulations,
                use_dirichlet=use_dirichlet,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps
            )
            if result == 1:
                wins_best += 1
            elif result == 2:
                wins_new += 1
            else:
                draws += 1

    total = wins_new + wins_best + draws
    winrate_new = wins_new / total if total > 0 else 0.0
    winrate_best = wins_best / total if total > 0 else 0.0
    draw_rate = draws / total if total > 0 else 0.0

    # Ensure csv folder exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Write (or append) to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "tag",
                "label_new",
                "label_best",
                "games",
                "wins_new",
                "wins_best",
                "draws",
                "winrate_new",
                "winrate_best",
                "draw_rate",
            ])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            tag if tag is not None else "",
            label_new,
            label_best,
            total,
            wins_new,
            wins_best,
            draws,
            f"{winrate_new:.6f}",
            f"{winrate_best:.6f}",
            f"{draw_rate:.6f}",
        ])

    # Colorful diagnostic printout
    header = (
        color("\n[HEAD-TO-HEAD] ", BOLD, CYAN)
        + color(f"{label_new} vs {label_best}", BOLD, WHITE)
    )
    print(header)

    print(
        "  "
        + color(f"{label_new} wins: ", DIM, WHITE)
        + color(str(wins_new), BOLD, GREEN)
        + color("  |  ", DIM, WHITE)
        + color(f"{label_best} wins: ", DIM, WHITE)
        + color(str(wins_best), BOLD, RED)
        + color("  |  ", DIM, WHITE)
        + color("Draws: ", DIM, WHITE)
        + color(str(draws), BOLD, YELLOW)
    )

    print(
        "  "
        + color(f"Win rate {label_new}: {winrate_new:.3f}", BOLD, GREEN)
        + color("   •   ", DIM, WHITE)
        + color(f"{label_best}: {winrate_best:.3f}", BOLD, RED)
        + color("   •   ", DIM, WHITE)
        + color(f"Draw rate: {draw_rate:.3f}", BOLD, YELLOW)
    )

    return wins_new, wins_best, draws

def evaluate_models(models, games_vs_prev=200, games_vs_random=400):
    """
    For each model i (i >= 1), evaluate it against model i-1:
      - model i as Player 1, model i-1 as Player 2
      - model i-1 as Player 1, model i as Player 2

    Then evaluate the final model against random (None) as both P1 and P2.
    All evaluations use e1=e2=0.0 (no randomness).
    Printed nicely with colors.
    """
    n = len(models)
    rows = []   # <-- collect results

    if n == 0:
        print(color("No models to evaluate.", BOLD, RED))
        return rows

    # --- successive model comparisons ---
    print(color("\n================ MODEL LADDER EVAL ================", BOLD, CYAN))
    for i in range(1, n):
        stronger = models[i]
        weaker   = models[i - 1]

        header = f"\n--- Model {i} vs Model {i-1} ---"
        print(color(header, BOLD, CYAN))

        # Model i as Player 1, model i-1 as Player 2
        w1_a, w2_a, d_a = _play_match(
            stronger, weaker,
            games=games_vs_prev // 2,
            e1=0.0, e2=0.0,
        )

        # Model i-1 as Player 1, model i as Player 2
        w1_b, w2_b, d_b = _play_match(
            weaker, stronger,
            games=games_vs_prev - games_vs_prev // 2,
            e1=0.0, e2=0.0,
        )

        # First call: w1_a = wins(stronger), w2_a = wins(weaker)
        # Second call: w1_b = wins(weaker),  w2_b = wins(stronger)
        wins_stronger = w1_a + w2_b
        wins_weaker   = w2_a + w1_b
        draws         = d_a + d_b
        total         = wins_stronger + wins_weaker + draws

        if total == 0:
            print(color("  (No games played?)", DIM, YELLOW))
            continue

        print(
            color("  Total games: ", DIM, WHITE)
            + color(str(total), BOLD, WHITE)
        )

        # Pick colors based on who did better
        if wins_stronger > wins_weaker:
            col_str = CYAN
            col_weak = RED
        elif wins_weaker > wins_stronger:
            col_str = RED
            col_weak = CYAN
        else:
            col_str = col_weak = YELLOW

        print(
            "  "
            + color(f"Model {i} wins: ", DIM, WHITE)
            + color(str(wins_stronger), BOLD, col_str)
            + color("  |  ", DIM, WHITE)
            + color(f"Model {i-1} wins: ", DIM, WHITE)
            + color(str(wins_weaker), BOLD, col_weak)
            + color("  |  draws: ", DIM, WHITE)
            + color(str(draws), BOLD, YELLOW)
        )

        print(
            "  "
            + color(f"Win rate M{i}: {wins_stronger/total:.3f}", col_str, BOLD)
            + color("   •   ", DIM, WHITE)
            + color(f"M{i-1}: {wins_weaker/total:.3f}", col_weak, BOLD)
            + color("   •   ", DIM, WHITE)
            + color(f"draw: {draws/total:.3f}", YELLOW, BOLD)
        )

    # --- final model vs random ---
    final_model = models[-1]
    print(color("\n============= FINAL MODEL vs RANDOM =============", BOLD, CYAN))

    # Final model as Player 1, random as Player 2
    w1_r1, w2_r1, d_r1 = _play_match(
        final_model, None,
        games=games_vs_random // 2,
        e1=0.0, e2=0.0,
    )

    # Random as Player 1, final model as Player 2
    w1_r2, w2_r2, d_r2 = _play_match(
        None, final_model,
        games=games_vs_random - games_vs_random // 2,
        e1=0.0, e2=0.0,
    )

    # First: w1_r1 = wins(final), w2_r1 = wins(random)
    # Second: w1_r2 = wins(random), w2_r2 = wins(final)
    wins_final   = w1_r1 + w2_r2
    wins_random  = w2_r1 + w1_r2
    draws_random = d_r1 + d_r2
    total_r      = wins_final + wins_random + draws_random

    if total_r == 0:
        print(color("  (No games played vs random?)", DIM, YELLOW))
        return

    print(
        color("  Total games vs random: ", DIM, WHITE)
        + color(str(total_r), BOLD, WHITE)
    )

    col_final = CYAN if wins_final >= wins_random else RED
    col_rand  = RED if wins_final >= wins_random else CYAN

    print(
        "  "
        + color("Final model wins: ", DIM, WHITE)
        + color(str(wins_final), BOLD, col_final)
        + color("  |  ", DIM, WHITE)
        + color("Random wins: ", DIM, WHITE)
        + color(str(wins_random), BOLD, col_rand)
        + color("  |  draws: ", DIM, WHITE)
        + color(str(draws_random), BOLD, YELLOW)
    )

    print(
        "  "
        + color(f"Final win rate: {wins_final/total_r:.3f}", col_final, BOLD)
        + color("   •   ", DIM, WHITE)
        + color(f"Random: {wins_random/total_r:.3f}", col_rand, BOLD)
        + color("   •   ", DIM, WHITE)
        + color(f"draw: {draws_random/total_r:.3f}", YELLOW, BOLD)
    )

    print(color("=================================================\n", BOLD, CYAN))

    return rows

def save_buffer(buffer, path="replay_buffer.pkl.gz"):
    """
    Save the replay buffer (list of (board, player, result)) to disk, compressed.
    """
    with gzip.open(path, "wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_buffer(path="replay_buffer.pkl.gz"):
    """
    Load the replay buffer from disk. Returns [] if file doesn't exist.
    """
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
    
