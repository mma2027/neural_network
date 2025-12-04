#!/usr/bin/env python3
"""
model_vs_model_viewer.py

Lets you choose two agents (Random, Heuristic MCTS, or NN+MCTS models)
and have them play a full game of Connect Four.

Game is recorded, then you can scrub through it with:
  ← (left arrow):  previous position
  → (right arrow): next position
  q               : quit viewer
"""

import os
import sys
import copy
import time

from neural_network import SimpleNN
from helper import (
    create_board,
    check_win,
    is_draw,
    drop_piece,
    print_board,
    color,
    BOLD,
    DIM,
    CYAN,
    GREEN,
    RED,
    YELLOW,
)
from baseline_agents import RandomAgent, HeuristicMCTS
from mcts import mcts_move


# ======================================================
# KEY HANDLING (CROSS PLATFORM)
# ======================================================

def _get_key_unix():
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        c1 = sys.stdin.read(1)
        if c1 in ("q", "Q"):
            return "QUIT"
        if c1 != "\x1b":
            return None
        c2 = sys.stdin.read(1)
        c3 = sys.stdin.read(1)
        if c2 == "[":
            if c3 == "C":
                return "RIGHT"
            elif c3 == "D":
                return "LEFT"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def _get_key_windows():
    import msvcrt
    c = msvcrt.getch()
    if c in (b"q", b"Q"):
        return "QUIT"
    if c in (b"\x00", b"\xe0"):
        c2 = msvcrt.getch()
        if c2 == b"K":
            return "LEFT"
        if c2 == b"M":
            return "RIGHT"
    return None


def get_key():
    if os.name == "nt":
        return _get_key_windows()
    return _get_key_unix()


# ======================================================
# AGENT WRAPPERS
# ======================================================

class AgentWrapper:
    def __init__(self, name):
        self.name = name
    def select_move(self, board, player):
        raise NotImplementedError


class RandomAgentWrapper(AgentWrapper):
    def __init__(self):
        super().__init__("Random")
        self.agent = RandomAgent()
    def select_move(self, board, player):
        move, _ = self.agent.select_move(board, player)
        return move


class HeuristicMCTSWrapper(AgentWrapper):
    def __init__(self, sims=200):
        super().__init__(f"HeuristicMCTS({sims})")
        self.agent = HeuristicMCTS(sims=sims)
    def select_move(self, board, player):
        move, _ = self.agent.select_move(board, player)
        return move


class NN_MCTS_Wrapper(AgentWrapper):
    def __init__(self, model_path, sims=400):
        name = f"NN+MCTS[{os.path.basename(model_path)}, sims={sims}]"
        super().__init__(name)
        self.model = SimpleNN.load(model_path)
        self.sims = sims
    def select_move(self, board, player):
        return mcts_move(
            board, player,
            model=self.model,
            num_simulations=self.sims,
            cpuct=1.4,
            dirichlet_noise=False
        )


# ======================================================
# SELECT AGENT
# ======================================================

def list_models(path="models"):
    if not os.path.isdir(path):
        return []
    out = [os.path.join(path, f)
           for f in os.listdir(path)
           if f.endswith(".npz")]
    return sorted(out)


def choose_agent(prompt, models_dir="models"):
    print()
    print(color(prompt, BOLD, CYAN))
    print(color("  [0] Random", DIM, CYAN))
    print(color("  [1] HeuristicMCTS", DIM, CYAN))

    model_files = list_models(models_dir)
    offset = 2

    if model_files:
        print(color("  [Models as NN+MCTS]:", BOLD, YELLOW))
        for i, m in enumerate(model_files):
            print(color(f"    [{i+offset}] {os.path.basename(m)}", DIM, CYAN))
    else:
        print(color("  No models found.", RED))

    while True:
        val = input(color("Choose index: ", BOLD, CYAN)).strip()
        if not val.isdigit():
            print(color("Not a number.", RED))
            continue
        idx = int(val)

        if idx == 0:
            return RandomAgentWrapper()
        if idx == 1:
            return HeuristicMCTSWrapper(200)

        midx = idx - offset
        if 0 <= midx < len(model_files):
            return NN_MCTS_Wrapper(model_files[midx])
        print(color("Invalid index.", RED))


# ======================================================
# GAME PLAY + RECORDING
# ======================================================

def play_recorded_game(a1, a2):
    from helper import get_valid_moves

    board = create_board()
    player = 1
    history = [(copy.deepcopy(board), player, None)]

    while True:
        if check_win(board, 1):
            return history, 1
        if check_win(board, 2):
            return history, 2
        if is_draw(board):
            return history, 0

        moves = get_valid_moves(board)
        if not moves:
            return history, 0

        agent = a1 if player == 1 else a2
        col = agent.select_move(board, player)
        if col not in moves:
            col = moves[0]

        drop_piece(board, col, player)
        nxt = 2 if player == 1 else 1
        history.append((copy.deepcopy(board), nxt, col))
        player = nxt


# ======================================================
# VIEWER
# ======================================================

def show_viewer(history, result, a1, a2):
    idx = 0
    n = len(history)

    while True:
        board, next_player, last_move = history[idx]

        print_board(board,
                    current_player=next_player if idx < n-1 else None)

        print(color(f"Position {idx+1}/{n}", BOLD, CYAN))
        print(color(f"P1 = {a1.name}", DIM, CYAN),
              color(f"P2 = {a2.name}", DIM, CYAN))

        if idx == n-1:
            if result == 1:
                print(color("P1 WINS", BOLD, GREEN))
            elif result == 2:
                print(color("P2 WINS", BOLD, RED))
            else:
                print(color("DRAW", BOLD, YELLOW))
        else:
            if idx > 0:
                _, prev_to_move, _ = history[idx-1]
                just_played = 2 if prev_to_move == 1 else 1
                print(color(f"Move #{idx}: Player {just_played} -> column {last_move+1}", BOLD, CYAN))

        print(color("Controls: ← back | → forward | q quit", DIM, CYAN))

        key = None
        while key is None:
            key = get_key()

        if key == "QUIT":
            break
        if key == "LEFT" and idx > 0:
            idx -= 1
        if key == "RIGHT" and idx < n-1:
            idx += 1


# ======================================================
# MAIN
# ======================================================

def main():
    print(color("Model vs Model Viewer", BOLD, CYAN))
    print(color("=======================", BOLD, CYAN))

    a1 = choose_agent("Choose agent for Player 1 (X)")
    a2 = choose_agent("Choose agent for Player 2 (O)")

    print(color("Simulating game...", DIM, CYAN))
    history, result = play_recorded_game(a1, a2)

    print(color("Simulation done — launching viewer!", BOLD, CYAN))
    time.sleep(0.5)

    show_viewer(history, result, a1, a2)


if __name__ == "__main__":
    main()
