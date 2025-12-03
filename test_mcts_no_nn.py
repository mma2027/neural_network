# test_mcts_no_nn.py

from mcts import MCTS
from helper import (
    ROWS, COLS,
    create_board,
    print_board,
    get_valid_moves,
    drop_piece,
    check_win,
    is_draw,
)


class HeuristicMCTS(MCTS):
    def __init__(self, cpuct: float = 1.4):
        # pass model=None, we won't use it
        super().__init__(model=None, cpuct=cpuct)

    def _nn_policy_value(self, board, player_to_move: int):
        """
        Override the NN call with a simple hand-coded heuristic.

        - If there is an immediate winning move for player_to_move, put
          all prior on those moves, value = +1.
        - Else if opponent has an immediate winning move next turn, put
          all prior on blocking moves, value = -1 (we're in trouble).
        - Else: uniform priors, value = 0.
        """
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            return [1.0 / COLS] * COLS, 0.0

        opp = 2 if player_to_move == 1 else 1

        # 1) check immediate wins for current player
        winning_moves = []
        for c in valid_moves:
            b2 = [row[:] for row in board]
            drop_piece(b2, c, player_to_move)
            if check_win(b2, player_to_move):
                winning_moves.append(c)

        if winning_moves:
            priors = [0.0] * COLS
            for c in winning_moves:
                priors[c] = 1.0 / len(winning_moves)
            v = 1.0  # clearly good for current player
            return priors, v

        # 2) check opponent immediate wins; blocking moves
        blocking_moves = []
        for c in valid_moves:
            b2 = [row[:] for row in board]
            drop_piece(b2, c, player_to_move)
            # now opponent to move: can they win?
            opp_can_win = False
            opp_moves = get_valid_moves(b2)
            for oc in opp_moves:
                b3 = [row[:] for row in b2]
                drop_piece(b3, oc, opp)
                if check_win(b3, opp):
                    opp_can_win = True
                    break
            if not opp_can_win:
                blocking_moves.append(c)

        if blocking_moves:
            priors = [0.0] * COLS
            for c in blocking_moves:
                priors[c] = 1.0 / len(blocking_moves)
            # position is still bad-ish (we were under threat),
            # but not insta-lost; say slight negative
            v = -0.3
            return priors, v

        # 3) otherwise: uniform, neutral-ish value
        priors = [0.0] * COLS
        for c in valid_moves:
            priors[c] = 1.0 / len(valid_moves)
        v = 0.0
        return priors, v


def demo_position(board, player, sims=200):
    print("\n=========== DEMO POSITION ===========")
    print_board(board)
    print(f"Player to move: {player}")

    from debug import analyze_mcts
    mcts = HeuristicMCTS(cpuct=1.4)
    analyze_mcts(board, player, mcts, sims)



if __name__ == "__main__":
    # 1) Immediate win for Player 1 in column 3
    from helper import create_board

    board_win = create_board()
    # three X's in col 3
    drop_piece(board_win, 3, 1)
    drop_piece(board_win, 3, 1)
    drop_piece(board_win, 3, 1)

    demo_position(board_win, player=1, sims=200)

    # 2) Must-block-or-die for Player 1 (three O's in col 3)
    board_block = create_board()
    drop_piece(board_block, 3, 2)
    drop_piece(board_block, 3, 2)
    drop_piece(board_block, 3, 2)

    demo_position(board_block, player=1, sims=200)
