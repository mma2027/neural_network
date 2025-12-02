# self_play.py

import random

from helper import (
    create_board, check_win, is_draw,
    get_valid_moves, drop_piece, print_board
)

from engines import engineR  # keep random engine
from mcts import mcts_move_with_policy


def self_play(model1, model2,
              e1=0, e2=0,
              record_states=True,
              display=True,
              simulations=200,
              record_policy=True):
    """
    Play a single self-play game between model1 (P1) and model2 (P2).

    Returns:
      (result, states)
    where:
      result: 1 if P1 wins, 2 if P2 wins, None for draw
      states: list of (board, player, pi) if record_states=True,
              or [] otherwise.
    """
    board = create_board()
    current_player = 1
    ai_states = []

    while True:
        # Terminal check
        if check_win(board, 1):
            if display:
                print_board(board)
            fixed = []
            for b, p, pi in ai_states:
                fixed.append((b, p, pi, 1))
            return 1, fixed

        if check_win(board, 2):
            if display:
                print_board(board)
            fixed = []
            for b, p, pi in ai_states:
                fixed.append((b, p, pi, 2))
            return 2, fixed

        if is_draw(board):
            if display:
                print_board(board)
            fixed = []
            for b, p, pi in ai_states:
                fixed.append((b, p, pi, 0.5))
            return None, fixed

        # Decide which model and epsilon to use
        if current_player == 1:
            model = model1
            eps = e1
        else:
            model = model2
            eps = e2

        moves = get_valid_moves(board)
        if not moves:
            # safety: treat as draw
            fixed = []
            for b, p, pi in ai_states:
                fixed.append((b, p, pi, 0.5))
            return None, fixed

        # Choose move + policy
        if model is None:
            # Pure random
            col = random.choice(moves)
            pi = [0.0] * 7
            for c in moves:
                pi[c] = 1.0 / len(moves)
        else:
            if random.random() < eps:
                # Random move for exploration
                col = random.choice(moves)
                pi = [0.0] * 7
                for c in moves:
                    pi[c] = 1.0 / len(moves)
            else:
                # Use MCTS with policy
                col, pi = mcts_move_with_policy(
                    board, current_player, model,
                    num_simulations=simulations
                )

        # Record state BEFORE applying the move
        if record_states:
            # deep copy board
            board_copy = [row[:] for row in board]
            ai_states.append((board_copy, current_player, pi))

        # Apply move
        drop_piece(board, col, current_player)

        # Switch player
        current_player = 2 if current_player == 1 else 1


def simulate(model1, model2,
             games=1,
             e1=0, e2=0,
             display=True,
             simulations=200,
             record_policy=True):
    """
    Run multiple self-play games and aggregate states.

    Returns:
      (all_states, win1, win2, draw)
    where all_states is a big list of (board, player, pi, result).
    """
    win1 = win2 = draw = 0
    all_states = []

    for _ in range(games):
        result, states = self_play(
            model1, model2,
            e1=e1, e2=e2,
            record_states=True,
            display=display,
            simulations=simulations,
            record_policy=record_policy,
        )

        if states is not None:
            all_states += states

        if result == 1:
            win1 += 1
        elif result == 2:
            win2 += 1
        else:
            draw += 1

    # print(win1, win2, draw)
    # print(all_states)
    return all_states, win1, win2, draw

# self_play(None, None)

# simulate(None, None, 1)