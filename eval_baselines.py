#!/usr/bin/env python3

import os
import random

from neural_network import SimpleNN
from mcts import mcts_move
from test_mcts_no_nn import HeuristicMCTS
from helper import (
    create_board,
    print_board,
    get_valid_moves,
    drop_piece,
    check_win,
    is_draw,
    color, BOLD, CYAN, YELLOW, GREEN, RED, MAGENTA, WHITE, DIM,
)
# ---------- Engine wrappers ----------

def make_nn_mcts_engine(model: SimpleNN, sims: int = 400, cpuct: float = 1.4):
    """
    Engine: uses your NN + MCTS policy to choose moves.
    Returns a function(board, player) -> column.
    """
    def engine(board, player):
        # Work on a copy because MCTS mutates the board inside.
        b_copy = [row[:] for row in board]
        col = mcts_move(
            b_copy,
            player,
            model,
            num_simulations=sims,
            cpuct=cpuct,
            dirichlet_noise=False,
        )
        return col
    engine.sims_per_move = sims
    engine.name = f"NN+MCTS({sims} sims)"
    return engine


def random_engine(board, player):
    """
    Pure random legal move engine.
    """
    moves = get_valid_moves(board)
    if not moves:
        return -1
    return random.choice(moves)

# attach metadata so our stats code can see it
random_engine.sims_per_move = 0
random_engine.name = "Random"


def make_heuristic_mcts_engine(sims: int = 400, cpuct: float = 1.4):
    """
    Engine: uses HeuristicMCTS (no NN, heuristic-only MCTS).
    """
    def engine(board, player):
        b_copy = [row[:] for row in board]
        mcts = HeuristicMCTS(cpuct=cpuct)
        col = mcts.search(
            b_copy,
            player_to_move=player,
            num_simulations=sims,
            dirichlet_noise=False,
        )
        return col
    engine.sims_per_move = sims
    engine.name = f"HeuristicMCTS({sims} sims)"
    return engine

# ---------- Core game loop ----------

def play_game(engineP1, engineP2, starting_player=1, display=False):
    """
    Play a single game where:
      - engineP1 controls player 1
      - engineP2 controls player 2

    Returns:
      result: 1 if P1 wins, 2 if P2 wins, 0 draw
      moves_P1: total moves made by player 1
      moves_P2: total moves made by player 2
      total_moves: moves_P1 + moves_P2
    """
    board = create_board()
    current_player = starting_player
    moves_P1 = 0
    moves_P2 = 0

    while True:
        if display:
            print_board(board, current_player=current_player)

        moves = get_valid_moves(board)
        if not moves:
            # should basically never happen without being draw
            if display:
                print(color("No legal moves, treat as draw.", BOLD, YELLOW))
            return 0, moves_P1, moves_P2, moves_P1 + moves_P2

        if current_player == 1:
            col = engineP1(board, 1)
            if col not in moves:
                # Safety: illegal → random fallback
                col = random.choice(moves)
            drop_piece(board, col, 1)
            moves_P1 += 1
        else:
            col = engineP2(board, 2)
            if col not in moves:
                col = random.choice(moves)
            drop_piece(board, col, 2)
            moves_P2 += 1

        # Terminal checks
        if check_win(board, 1):
            if display:
                print_board(board, current_player=None)
                print(color("Player 1 WINS", BOLD, GREEN))
            return 1, moves_P1, moves_P2, moves_P1 + moves_P2

        if check_win(board, 2):
            if display:
                print_board(board, current_player=None)
                print(color("Player 2 WINS", BOLD, RED))
            return 2, moves_P1, moves_P2, moves_P1 + moves_P2

        if is_draw(board):
            if display:
                print_board(board, current_player=None)
                print(color("Draw.", BOLD, YELLOW))
            return 0, moves_P1, moves_P2, moves_P1 + moves_P2

        current_player = 2 if current_player == 1 else 1

# ---------- Match runner with stats ----------

def run_matchup(nameA, engineA, nameB, engineB,
                games=100, swap_colors=True, display_every=0):
    """
    Run many games between engineA and engineB, optionally swapping colors.

    We track:
      - wins / losses / draws
      - moves by each engine
      - sims used (moves * sims_per_move)
    """
    wins_A = wins_B = draws = 0
    moves_A = moves_B = 0
    total_moves = 0

    sims_per_move_A = getattr(engineA, "sims_per_move", 0)
    sims_per_move_B = getattr(engineB, "sims_per_move", 0)

    for g in range(games):
        swapped = swap_colors and (g % 2 == 1)

        if swapped:
            # B is P1, A is P2
            if display_every and g % display_every == 0:
                print(color(
                    f"\nGame {g+1}/{games}: {nameB} (P1) vs {nameA} (P2)",
                    BOLD, CYAN
                ))
            result, mP1, mP2, mTotal = play_game(
                engineB, engineA,
                starting_player=1,
                display=(display_every and g % display_every == 0)
            )

            # Map P1/P2 to A/B
            moves_B += mP1
            moves_A += mP2
            if result == 1:
                wins_B += 1
            elif result == 2:
                wins_A += 1
            else:
                draws += 1

        else:
            # A is P1, B is P2
            if display_every and g % display_every == 0:
                print(color(
                    f"\nGame {g+1}/{games}: {nameA} (P1) vs {nameB} (P2)",
                    BOLD, CYAN
                ))
            result, mP1, mP2, mTotal = play_game(
                engineA, engineB,
                starting_player=1,
                display=(display_every and g % display_every == 0)
            )

            moves_A += mP1
            moves_B += mP2
            if result == 1:
                wins_A += 1
            elif result == 2:
                wins_B += 1
            else:
                draws += 1

        total_moves += mTotal

    total_games = wins_A + wins_B + draws
    score_A = (wins_A + 0.5 * draws) / total_games if total_games else 0.0
    score_B = (wins_B + 0.5 * draws) / total_games if total_games else 0.0

    total_sims_A = moves_A * sims_per_move_A
    total_sims_B = moves_B * sims_per_move_B

    avg_sims_per_move_A = total_sims_A / moves_A if moves_A > 0 else 0.0
    avg_sims_per_move_B = total_sims_B / moves_B if moves_B > 0 else 0.0

    print(color("\n======================================", BOLD, MAGENTA))
    print(color(f"  MATCH RESULT: {nameA} vs {nameB}", BOLD, MAGENTA))
    print(color("======================================", BOLD, MAGENTA))

    print(
        color(f"{nameA}: {wins_A} wins  ", BOLD, GREEN) +
        color(f"{nameB}: {wins_B} wins  ", BOLD, RED) +
        color(f"Draws: {draws}", BOLD, YELLOW)
    )

    print(
        color(f"\nScore (win + 0.5·draw): ", BOLD, WHITE) +
        color(f"{nameA}: {score_A:.3f}   ", CYAN) +
        color(f"{nameB}: {score_B:.3f}", CYAN)
    )

    print(color("\n--- Search statistics ---", BOLD, WHITE))
    print(
        color(f"{nameA} moves: {moves_A}   ", DIM, WHITE) +
        color(f"{nameB} moves: {moves_B}", DIM, WHITE)
    )
    print(
        color(f"{nameA} total sims: {total_sims_A}   ", CYAN) +
        color(f"{nameB} total sims: {total_sims_B}", CYAN)
    )
    print(
        color(f"{nameA} avg sims/move: {avg_sims_per_move_A:.1f}   ", CYAN) +
        color(f"{nameB} avg sims/move: {avg_sims_per_move_B:.1f}", CYAN)
    )
    print(
        color(f"Total moves across all games: {total_moves}", BOLD, MAGENTA)
    )
    print(color("======================================\n", BOLD, MAGENTA))

    return {
        "wins_A": wins_A,
        "wins_B": wins_B,
        "draws": draws,
        "score_A": score_A,
        "score_B": score_B,
        "moves_A": moves_A,
        "moves_B": moves_B,
        "total_moves": total_moves,
        "total_sims_A": total_sims_A,
        "total_sims_B": total_sims_B,
    }

# ---------- Main ----------

def load_final_model(path="models/model_final.npz") -> SimpleNN:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")
    print(color(f"Loading model from {path}", BOLD, CYAN))
    model = SimpleNN.load(path)
    print(color("Model loaded.\n", BOLD, CYAN))
    return model


def main():
    model = load_final_model("models/model_final.npz")

    # Build engines
    nn_sims = 400
    heur_sims = 400

    nn_mcts = make_nn_mcts_engine(model, sims=nn_sims, cpuct=1.4)
    rand_eng = random_engine
    heur_mcts = make_heuristic_mcts_engine(sims=heur_sims, cpuct=1.4)

    # 1) NN+MCTS vs Random
    run_matchup("NN+MCTS", nn_mcts, "Random", rand_eng,
                games=100, swap_colors=True, display_every=0)

    # 2) NN+MCTS vs HeuristicMCTS
    run_matchup("NN+MCTS", nn_mcts, "HeuristicMCTS", heur_mcts,
                games=100, swap_colors=True, display_every=0)


if __name__ == "__main__":
    main()
