import numpy as np
import os
import csv
from neural_network import SimpleNN
from helper import (
    board_to_vector, evaluate_models,
    color, BOLD, CYAN, YELLOW, GREEN, RED, MAGENTA,
    load_buffer, save_buffer
)
from self_play import simulate
from engines import engineR  # not strictly needed but fine

def train_on_buffer_az(model, buffer, epochs=4, lr=0.005,
                       shuffle=True, verbose=True):
    """
    AlphaZero-style training on a replay buffer of:
        (board, player, pi, result)
    """
    N = len(buffer)
    if N == 0:
        return

    for epoch in range(epochs):
        indices = np.arange(N)
        if shuffle:
            np.random.shuffle(indices)

        total_bce = 0.0
        total_ce = 0.0

        for idx in indices:
            board, player, pi, result = buffer[idx]

            x = board_to_vector(board, player)

            # Value label: 1 (win for player), 0 (loss), 0.5 (draw)
            if result == 0.5:
                y_val = 0.5
            else:
                y_val = 1.0 if player == result else 0.0

            # Ensure pi is a proper distribution
            pi_arr = np.asarray(pi, dtype=float)
            s = pi_arr.sum()
            if s <= 0.0 or not np.isfinite(s):
                pi_arr = np.ones_like(pi_arr) / len(pi_arr)
            else:
                pi_arr = pi_arr / s

            # Forward
            p_pred, v_pred = model.forward_policy_value(x)

            # Losses for logging only
            v = float(v_pred)
            eps_num = 1e-7
            v_clamped = max(min(v, 1.0 - eps_num), eps_num)
            bce_loss = -(y_val * np.log(v_clamped) +
                         (1.0 - y_val) * np.log(1.0 - v_clamped))
            total_bce += bce_loss

            p_pred = np.asarray(p_pred, dtype=float)
            ce_loss = -np.sum(pi_arr * np.log(np.clip(p_pred, eps_num, 1.0)))
            total_ce += ce_loss

            # Backward update on combined loss
            model.backward_az(y_val, pi_arr, lr=lr,
                              value_weight=1.0, policy_weight=1.0)

        avg_bce = total_bce / N
        avg_ce = total_ce / N

        if verbose:
            print(
                color("    ▸ ", BOLD, CYAN) +
                color(f"Epoch {epoch+1}/{epochs} ", BOLD, MAGENTA) +
                color(f"Value={avg_bce:.4f} ", GREEN) +
                color(f"Policy={avg_ce:.4f}", YELLOW)
            )

REPLAY_PATH = "replay_buffer_az.pkl.gz"
MAX_BUFFER_SIZE = 100000

model = SimpleNN(hidden=256)
models = []


iterations     = 40
games_initial  = 1000    # random warmup
games_per_iter = 1000    # self-play games per generation

train_epochs   = 6       # passes over buffer each gen
train_lr       = 0.005   # smaller than 0.01 → more stable

eps = 0.5
eps_min = 0.05
eps_decay = 0.95

train_sims = 50

buffer = load_buffer(REPLAY_PATH)

train_log_path = "csv/training_log.csv"
if not os.path.exists(train_log_path):
    with open(train_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration",
            "wins_p1",
            "wins_p2",
            "draws",
            "total_games",
            "winrate_p1",
            "winrate_p2",
            "draw_rate",
            "epsilon"
        ])

if not buffer:
    # Initial warmup with random-ish play
    print(color("No replay buffer found. Generating warmup games...", BOLD, YELLOW))
    data_states, win1, win2, draw = simulate(
        None, None,
        games=games_initial,
        e1=0.0, e2=0.0,
        display=False,
        simulations=10,      # irrelevant when model=None
        record_policy=True,
    )
    buffer = list(data_states)
    print(color(f"Warmup done: P1={win1}, P2={win2}, D={draw}", BOLD, CYAN))


for t in range(iterations):
    print(color(f"\n====== TRAINING ITERATION {t} ======", BOLD, CYAN))
    print(color(f"Exploration ε = {eps:.3f}", BOLD, YELLOW))

    # Train on current buffer
    train_on_buffer_az(model, buffer, epochs=train_epochs, lr=train_lr)

    # Save snapshot
    models.append(model.clone())

    # Generate new self-play games with current model
    new_states, win1, win2, draw = simulate(
        model, model,
        games=games_per_iter,
        e1=eps, e2=eps,
        display=False,
        simulations=train_sims,
        record_policy=True,
    )

    print(
        color("Self-Play Results → ", BOLD, CYAN) +
        color(f"P1: {win1}  ", GREEN) +
        color(f"P2: {win2}  ", RED) +
        color(f"Draws: {draw}", YELLOW)
    )

    total_games = win1 + win2 + draw
    with open(train_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            iterations,
            win1,
            win2,
            draw,
            total_games,
            win1 / total_games if total_games else 0,
            win2 / total_games if total_games else 0,
            draw / total_games if total_games else 0,
            eps
        ])
        
    buffer += new_states
    if len(buffer) > MAX_BUFFER_SIZE:
        buffer = buffer[-MAX_BUFFER_SIZE:]


    print(color(f"Replay Buffer Size: {len(buffer)} / {MAX_BUFFER_SIZE}", BOLD, MAGENTA))
    
    eps = max(eps * eps_decay, eps_min)

    save_buffer(buffer, REPLAY_PATH)

# Save models
for idx, m in enumerate(models):
    m.save(f"models/model_gen_{idx}.npz")

final_model = models[-1]
final_model.save("models/model_final.npz")

print(color("\n============================================", BOLD, CYAN))
print(color("       TRAINING COMPLETE — MODEL SAVED       ", BOLD, GREEN))
print(color("============================================\n", BOLD, CYAN))

# Evaluate ladder
rows = evaluate_models(models, games_vs_prev=100, games_vs_random=100)

csv_path = "csv/model_eval_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "kind",
        "model_i",
        "model_j",
        "wins_i",
        "wins_j",
        "draws",
        "total",
        "win_rate_i",
        "win_rate_j",
    ])
    for r in rows:
        writer.writerow([
            r["kind"],
            r["model_i"],
            r["model_j"],
            r["wins_i"],
            r["wins_j"],
            r["draws"],
            r["total"],
            f"{r['win_rate_i']:.6f}",
            f"{r['win_rate_j']:.6f}",
        ])

print(color(f"Evaluation summary saved to {csv_path}", BOLD, CYAN))