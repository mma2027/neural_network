# debug.py
# Simple all-in-one debugging tool for NN + MCTS
# Edit the CONFIG section at the top and run `python3 debug.py`

from helper import (
    ROWS, COLS,
    create_board,
    print_board,
    board_to_vector,
)
from neural_network import SimpleNN
from mcts import MCTS, MCTSNode

import math


# ================================================================
# ======================= CONFIG SECTION ==========================
# ================================================================

MODEL_PATH = "models/model_final.npz"     # which model to load

PLAYER_TO_MOVE = 1                        # 1 or 2

# Hardcode the board position here.
# Format: 6 rows, '/' separated, '.' empty, '1' player1, '2' player2
# Top row first.
BOARD_STRING = (
    "......./"
    "......./"
    "......./"
    "...2.../"
    "...2.../"
    "...2..."
)

NUM_SIMULATIONS = 200
CPUCT = 1.4
USE_DIRICHLET = False
DIR_ALPHA = 0.3
DIR_EPS = 0.25

# ================================================================
# ======================= END CONFIG =============================
# ================================================================


def parse_board_string(s):
    """Parse BOARD_STRING into a board matrix."""
    rows = s.strip().split('/')
    if len(rows) != ROWS:
        raise ValueError(f"Expected {ROWS} rows, got {len(rows)}")

    board = create_board()
    for r, row_str in enumerate(rows):
        if len(row_str) != COLS:
            raise ValueError(
                f"Row {r} has length {len(row_str)}, expected {COLS}"
            )
        for c, ch in enumerate(row_str):
            if ch in ('.', '0'):
                board[r][c] = 0
            elif ch == '1':
                board[r][c] = 1
            elif ch == '2':
                board[r][c] = 2
            else:
                raise ValueError(f"Invalid char '{ch}' in board string.")
    return board


# ================================================================
# ====================== NN ANALYSIS =============================
# ================================================================

def analyze_nn(board, player, model):
    print("\n================ NN ANALYSIS ================")
    print_board(board)
    print(f"Player to move: {player}")

    x = board_to_vector(board, player)
    policy, v_prob = model.forward_policy_value(x)
    v_prob = float(v_prob)
    v_mcts = 2 * v_prob - 1  # Your mapping

    print(f"\nNN VALUE (P(win for player {player})): {v_prob:.4f}")
    print(f"NN VALUE mapped to [-1,1]:           {v_mcts:.4f}")

    print("\nPolicy (raw softmax output):")
    for c in range(COLS):
        print(f"  col {c}: {policy[c]:.4f}")

    print("\nPolicy sorted (desc):")
    sorted_pi = sorted(
        [(c, policy[c]) for c in range(COLS)],
        key=lambda x: x[1],
        reverse=True,
    )
    for c, p in sorted_pi:
        print(f"  col {c}: {p:.4f}")


# ================================================================
# ====================== MCTS ANALYSIS ===========================
# ================================================================

def analyze_mcts(board, player, mcts, num_simulations=200):
    """
    Analyze an MCTS instance on the given board.

    mcts: an object with _simulate(...) and cpuct attribute (MCTS or HeuristicMCTS)
    num_simulations: how many simulations to run from the root
    """
    print("\n================ MCTS ANALYSIS ================")
    print(f"Simulations: {num_simulations}, c_puct={mcts.cpuct}")

    root = MCTSNode(prior=1.0)

    # Run MCTS manually so we can inspect the root node
    for _ in range(num_simulations):
        b = [row[:] for row in board]
        mcts._simulate(
            b,
            player_to_move=player,
            node=root,
            is_root=True,
            dirichlet_noise=USE_DIRICHLET,
            dirichlet_alpha=DIR_ALPHA,
            dirichlet_eps=DIR_EPS,
        )

    if not root.children:
        print("No valid moves from root (terminal board?).")
        return

    total_visits = sum(child.N for child in root.children.values())
    parent_visits = root.N + 1e-8

    # Collect info
    info = []
    for mv, child in root.children.items():
        N = child.N
        Q = child.Q
        P = child.P
        q_eff = -Q  # value from current player's perspective
        U = q_eff + mcts.cpuct * P * math.sqrt(parent_visits) / (1 + N)
        pi = N / total_visits if total_visits > 0 else 0.0
        info.append((mv, N, Q, q_eff, P, U, pi))

    # Best move
    best = max(info, key=lambda t: t[1])[0]
    print(f"\nBest move by visits: col {best} (human {best+1})")

    print("\nMove stats (sorted by visits):")
    info.sort(key=lambda t: t[1], reverse=True)
    for mv, N, Q_raw, Q_eff, P, U, pi in info:
        print(
            f"  col {mv}: "
            f"N={N}, Q_raw={Q_raw:.4f}, Q_eff={Q_eff:.4f}, "
            f"P(prior)={P:.4f}, PUCT={U:.4f}, pi_vis={pi:.4f}"
        )

    print("\nVisit policy:")
    for mv in range(COLS):
        child = root.children.get(mv)
        if child:
            print(f"  col {mv}: {child.N/total_visits:.4f}")
        else:
            print(f"  col {mv}: 0.0000")



# ================================================================
# =========================== MAIN ===============================
# ================================================================

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = SimpleNN.load(MODEL_PATH)

    board = parse_board_string(BOARD_STRING)

    analyze_nn(board, PLAYER_TO_MOVE, model)

    # create a normal NN-based MCTS instance for debugging
    mcts = MCTS(model=model, cpuct=CPUCT)
    analyze_mcts(board, PLAYER_TO_MOVE, mcts, NUM_SIMULATIONS)


if __name__ == "__main__":
    main()
