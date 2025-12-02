#!/usr/bin/env python3

import sys
import os

from neural_network import SimpleNN
from helper import (
    create_board,
    print_board,
    get_valid_moves,
    drop_piece,
    check_win,
    is_draw,
    predict_move,
    color,
    BOLD,
    DIM,
    RED,
    YELLOW,
    CYAN,
    WHITE,
)


def load_model(path: str) -> SimpleNN:
    """
    Load a saved model from an .npz file using SimpleNN.load(path).
    """
    if not os.path.exists(path):
        print(color(f"Model file '{path}' not found.", BOLD, RED))
        sys.exit(1)

    print(color(f"Loading model from {path}", BOLD, CYAN))
    model = SimpleNN.load(path)
    print(color("Model loaded successfully.\n", BOLD, CYAN))
    return model


def choose_side() -> int:
    """
    Let the user choose whether to play as Player 1 (X) or Player 2 (O).
    Returns 1 or 2.
    """
    while True:
        print(
            color("Choose your side:", BOLD, WHITE),
            color(" [1] Player 1 (X)  [2] Player 2 (O)", DIM, WHITE),
        )
        choice = input(color("Enter 1 or 2: ", BOLD, CYAN)).strip()
        if choice in ("1", "2"):
            return int(choice)
        print(color("Invalid choice. Please enter 1 or 2.\n", BOLD, RED))


def get_human_move(board) -> int:
    """
    Prompt the human for a column (1–7) and return the 0-based column index.
    """
    valid_moves = get_valid_moves(board)
    valid_display = [c + 1 for c in valid_moves]

    while True:
        print(color(f"Valid moves: {valid_display}", DIM, WHITE))
        raw = input(color("Your move (1-7): ", BOLD, CYAN)).strip()
        if not raw.isdigit():
            print(color("Please enter a number between 1 and 7.\n", BOLD, RED))
            continue

        col = int(raw) - 1
        if col in valid_moves:
            return col
        print(color("That column is not valid or is full. Try again.\n", BOLD, RED))


def get_ai_move(board, player, model: SimpleNN) -> int:
    """
    Use the trained model to pick the best move for `player`.
    """
    col = predict_move(board, player, model)
    return col


def play_game(model: SimpleNN, human_player: int):
    """
    Play a single game of Connect Four: human vs model.
    human_player is 1 or 2.
    """
    board = create_board()
    current_player = 1

    print(color("\nStarting a new game!", BOLD, CYAN))
    print(
        color("You are ", DIM, WHITE)
        + (
            color("Player 1 (X)", BOLD, RED)
            if human_player == 1
            else color("Player 2 (O)", BOLD, YELLOW)
        )
    )
    print()

    while True:
        print_board(board, current_player=current_player)

        if current_player == human_player:
            print(color("Your turn.", BOLD, WHITE))
            col = get_human_move(board)
        else:
            print(color("AI is thinking...", DIM, WHITE))
            col = get_ai_move(board, current_player, model)
            print(color(f"AI plays column {col + 1}", BOLD, YELLOW))

        drop_piece(board, col, current_player)

        # Check terminal states
        if check_win(board, current_player):
            print_board(board, current_player=current_player)
            if current_player == human_player:
                print(color("You WIN! 🎉", BOLD, CYAN))
            else:
                print(color("AI wins. Better luck next time!", BOLD, RED))
            break

        if is_draw(board):
            print_board(board, current_player=None)
            print(color("It's a draw.", BOLD, YELLOW))
            break

        # Switch player
        current_player = 2 if current_player == 1 else 1


def choose_model_from_folder(models_dir: str = "models") -> str:
    """
    If no model path is passed on the command line, look in `models/`
    for .npz files and let the user select one by number or name.
    """
    if not os.path.isdir(models_dir):
        print(
            color(
                f"No '{models_dir}/' directory found. "
                "Falling back to 'model_final.npz'.",
                BOLD,
                YELLOW,
            )
        )
        return "model_final.npz"

    files = [
        f for f in os.listdir(models_dir)
        if f.endswith(".npz") and os.path.isfile(os.path.join(models_dir, f))
    ]

    if not files:
        print(
            color(
                f"No .npz files found in '{models_dir}/'. "
                "Falling back to 'model_final.npz'.",
                BOLD,
                YELLOW,
            )
        )
        return "model_final.npz"

    files.sort()

    print(color("\nAvailable models in 'models/':", BOLD, WHITE))
    for idx, fname in enumerate(files):
        print(
            "  "
            + color(f"[{idx}]", DIM, CYAN)
            + " "
            + color(fname, BOLD, WHITE)
        )

    print(
        color(
            "\nSelect a model by number or type a filename "
            "(press Enter for [0]):",
            DIM,
            WHITE,
        )
    )

    while True:
        choice = input(color("Model choice: ", BOLD, CYAN)).strip()

        # Default: index 0
        if choice == "":
            chosen = files[0]
            break

        # If it's a digit, treat as index
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(files):
                chosen = files[idx]
                break
            else:
                print(color("Invalid index. Try again.", BOLD, RED))
                continue

        # Otherwise, treat as filename (allow partial convenience)
        # First check exact match
        if choice in files:
            chosen = choice
            break

        # Try prefix match
        matches = [f for f in files if f.startswith(choice)]
        if len(matches) == 1:
            chosen = matches[0]
            print(
                color(
                    f"Interpreting '{choice}' as '{chosen}'.",
                    DIM,
                    YELLOW,
                )
            )
            break
        elif len(matches) > 1:
            print(
                color(
                    "Ambiguous prefix, matches multiple files:",
                    BOLD,
                    YELLOW,
                )
            )
            for m in matches:
                print("   " + color(m, DIM, WHITE))
            continue
        else:
            print(color("No such model name or prefix. Try again.", BOLD, RED))

    full_path = os.path.join(models_dir, chosen)
    print(color(f"\nSelected model: {full_path}\n", BOLD, CYAN))
    return full_path


def main():
    # Model path: from CLI arg or from models/ folder, or default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = choose_model_from_folder(models_dir="models")

    model = load_model(model_path)
    human_side = choose_side()

    while True:
        play_game(model, human_side)
        again = input(color("\nPlay again? (y/n): ", BOLD, CYAN)).strip().lower()
        if again not in ("y", "yes"):
            print(color("Thanks for playing!", BOLD, WHITE))
            break


if __name__ == "__main__":
    main()
