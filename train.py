import numpy as np
import random
from neural_network import SimpleNN
from helper import board_to_vector, evaluate_models, color, BOLD, DIM, RED, YELLOW, CYAN, WHITE, load_buffer, save_buffer
from engines import engineR, engine, explore
from self_play import(
    simulate
)

def train_on_dataset(model, X, y, epochs=10, lr=0.01, shuffle=True, verbose=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = X.shape[0]

    # For tracking best epoch and EMA (smoothed) losses
    best_bce = float("inf")
    ema_bce = None
    ema_mse = None
    ema_alpha = 0.1  # 10% new, 90% old

    for epoch in range(epochs):
        indices = np.arange(N)
        if shuffle:
            np.random.shuffle(indices)

        total_bce = 0.0
        total_mse = 0.0

        for i in indices:
            x_i = X[i]
            y_i = y[i]

            # Forward pass
            pred = model.forward(x_i)

            # Clamp prediction a bit to avoid log(0) in BCE
            eps = 1e-7
            p = float(np.clip(pred, eps, 1.0 - eps))

            # Binary cross-entropy (this matches your gradient dz2 = a2 - y)
            # Works fine even when y_i is 0.5 (draws)
            bce_loss = -(y_i * np.log(p) + (1.0 - y_i) * np.log(1.0 - p))

            # MSE-ish loss (your old one), mainly for intuition
            mse_loss = 0.5 * (pred - y_i) ** 2

            total_bce += bce_loss
            total_mse += mse_loss

            # Backward pass + parameter update
            model.backward(y_i, lr=lr)

        avg_bce = total_bce / N
        avg_mse = total_mse / N

        # Update EMA
        if ema_bce is None:
            ema_bce = avg_bce
            ema_mse = avg_mse
        else:
            ema_bce = (1 - ema_alpha) * ema_bce + ema_alpha * avg_bce
            ema_mse = (1 - ema_alpha) * ema_mse + ema_alpha * avg_mse

        # Track best epoch by BCE
        improved = avg_bce < best_bce
        if improved:
            best_bce = avg_bce

        if verbose:
            # Choose color: cyan normally, yellow if worse than best, greenish (cyan) with star if best
            if improved:
                line_color = CYAN
                star = " ★"
            else:
                line_color = YELLOW
                star = ""

            msg = (
                f"Epoch {epoch+1}/{epochs}"
                f" | BCE: {avg_bce:.4f} (EMA: {ema_bce:.4f})"
                f" | MSE: {avg_mse:.4f} (EMA: {ema_mse:.4f})"
            )

            print(color(msg + star, BOLD, line_color))


def train(model, dataset, epochs=10, lr=0.01):
    X = []
    y = []

    for board, player, result in dataset:
        X.append(board_to_vector(board, player))

        if result == 0.5:
            # draw → neutral label
            label = 0.5
        else:
            # result is 1 or 2 (the winner)
            label = 1.0 if player == result else 0.0

        y.append(label)

    train_on_dataset(model, X, y, epochs=epochs, lr=lr)

REPLAY_PATH = "replay_buffer.pkl.gz"
MAX_BUFFER_SIZE = 100000
iteration = 20
models = []

eps = 0.8
eps_min = 0.05
eps_decay = 0.95

buffer = load_buffer(REPLAY_PATH)

if not buffer:
    # if none, do initial random warmup
    data_states, win1, win2, draw = simulate(None, None, games=5000, display=True)
    buffer = list(data_states)

for t in range(iteration):
    print("Training model", t, "with eps =", eps)
    if t == 0:
        model = SimpleNN(hidden=256)
    else:
        # reuse previous model instead of making a fresh one
        model = models[-1]

    train(model, buffer, epochs=15, lr=0.01)
    models.append(model.clone())

    new_states, win1, win2, draw = simulate(model, model, games=3000, e1=eps, e2=eps)
    # data = new_states  # if you want to *only* use latest states
    buffer += new_states
    if len(buffer) > MAX_BUFFER_SIZE:
        buffer = buffer[-MAX_BUFFER_SIZE:]
    eps = max(eps * eps_decay, eps_min)

save_buffer(buffer, REPLAY_PATH)

for idx, m in enumerate(models):
    m.save(f"models/model_gen_{idx}.npz")

final_model = models[-1]
final_model.save("models/model_final.npz")

evaluate_models(models, games_vs_prev=1000, games_vs_random=4000)