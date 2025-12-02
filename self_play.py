import random

from helper import(
    create_board, check_win, is_draw, get_valid_moves, drop_piece, print_board
)

from engines import(
    engineR, engine
)

def self_play(model1, model2, e1=0, e2=0, record_states=True, display=True, simulations=200):
    from engines import engine
    board = create_board()
    current_player = 1
    ai_states = []

    while True:
        # Terminal check first to avoid calling moves on a full/finished board
        if check_win(board, 1):
            if display:
                print_board(board)
            fixed = []
            for _ in ai_states:
                fixed.append(_ + (1,))
            return 1, fixed
        if check_win(board, 2):
            if display:
                print_board(board)
            fixed = []
            for _ in ai_states:
                fixed.append(_ + (2,))
            return 2, fixed
        if is_draw(board):
            if display:
                print_board(board)
            fixed = []
            for _ in ai_states:
                fixed.append(_ + (0.5,))  # draw label
            return None, fixed

        # Record state whenever it's our chosen AI player's turn
        if record_states:
            ai_states.append(([row[:] for row in board], current_player))  # deep copy

        moves = get_valid_moves(board)
        if not moves:
            # Safety net: treat as draw
            return None, ai_states
        if current_player == 1:
            col = engine(board, current_player, model1, e1, simulations=simulations)
        else:
            col = engine(board, current_player, model2, e2, simulations=simulations)
        drop_piece(board, col, current_player)

        # Switch player
        current_player = 2 if current_player == 1 else 1

def simulate(model1, model2, games=1, e1=0, e2=0, display=True, simulations=200):
    win1 = 0
    win2 = 0
    draw = 0
    ai_states = []
    for _ in range(games):
        result, states = self_play(model1, model2, e1, e2, record_states=True, display=display, simulations=simulations)
        # print(len(states))
        if states is not None:
            ai_states += states
        if result == 1:
            win1 += 1
        elif result == 2:
            win2 += 1
        else:
            draw += 1
    print(win1, win2, draw)
    # print(ai_states)
    return ai_states, win1, win2, draw
# self_play(None, None)

# simulate(None, None, 1)