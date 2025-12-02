import random

from helper import(
    get_valid_moves, predict_move
)

def engineR(board, player):
    moves = get_valid_moves(board)
    return random.choice(moves)

def engine(board, player, model=None, e=0):
    if model is None or random.random() < e:
        return engineR(board, player)
    return predict_move(board, player, model)

def explore(model1, model2, e):
    if random.random() < e:
        return model2
    else:
        return model1