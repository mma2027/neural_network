# baseline_agents.py

import random
import numpy as np
from helper import get_valid_moves, drop_piece, check_win, is_draw, create_board
from mcts import MCTS

# -------------------------------
# 1. RANDOM AGENT
# -------------------------------
class RandomAgent:
    def select_move(self, board, player):
        moves = get_valid_moves(board)
        return random.choice(moves), 0  # 0 simulations used


# -------------------------------
# 2. PURE HEURISTIC MCTS (NO NN)
# -------------------------------
class HeuristicMCTS:
    """
    Pure MCTS using rollouts, no NN.
    """
    def __init__(self, sims=200):
        self.sims = sims

    def rollout(self, board, player):
        """Play random moves to terminal state."""
        b = [row[:] for row in board]
        p = player

        while True:
            if check_win(b, 1): return 1
            if check_win(b, 2): return 2
            if is_draw(b): return 0.5

            moves = get_valid_moves(b)
            move = random.choice(moves)
            drop_piece(b, move, p)
            p = 1 if p == 2 else 2

    def select_move(self, board, player):
        moves = get_valid_moves(board)
        wins = {m: 0 for m in moves}
        plays = {m: 0 for m in moves}

        for _ in range(self.sims):
            move = random.choice(moves)
            b = [row[:] for row in board]
            drop_piece(b, move, player)
            result = self.rollout(b, 1 if player == 2 else 2)
            plays[move] += 1

            if result == player:
                wins[move] += 1
            elif result == 0.5:
                wins[move] += 0.5

        best = max(moves, key=lambda m: wins[m] / plays[m])
        return best, self.sims
