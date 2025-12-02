import time
import numpy as np

from self_play import simulate
from neural_network import SimpleNN
from helper import evaluate_models

# ---- match these to your current train.py hyperparams ----
ITERATIONS       = 20       # number of training iterations
GAMES_INITIAL    = 3000     # warmup random games
GAMES_PER_ITER   = 100     # self-play per iteration

TRAIN_SIMS       = 30       # MCTS sims during training
EPS              = 0.8      # starting epsilon

TRAIN_EPOCHS     = 4        # epochs per iteration
REPLAY_MAX_SIZE  = 100_000  # from train.py
INPUT_SIZE       = 84
HIDDEN_SIZE      = 256
# ----------------------------------------------------------


def time_random_games(num_games=200):
    """
    Time pure-random self-play (model=None) to estimate warmup cost.
    """
    print(f"Timing {num_games} RANDOM games (warmup-style)...")
    t0 = time.perf_counter()
    _, w1, w2, d = simulate(
        None, None,
        games=num_games,
        e1=0.0,
        e2=0.0,
        display=False,
    )
    t1 = time.perf_counter()

    total = t1 - t0
    per_game = total / float(num_games)
    print(f"  Took {total:.3f}s total  ->  {per_game:.4f}s / game")
    print(f"  Results: P1={w1}, P2={w2}, D={d}")
    return per_game


def time_mcts_games(num_games=200):
    """
    Time self-play with your NN + MCTS (same sims & eps as training).
    """
    print(f"\nTiming {num_games} MCTS games (training settings)...")
    model = SimpleNN(hidden=HIDDEN_SIZE)

    t0 = time.perf_counter()
    _, w1, w2, d = simulate(
        model, model,
        games=num_games,
        e1=EPS,
        e2=EPS,
        display=False,
        simulations=TRAIN_SIMS,
    )
    t1 = time.perf_counter()

    total = t1 - t0
    per_game = total / float(num_games)
    print(f"  Took {total:.3f}s total  ->  {per_game:.4f}s / game")
    print(f"  Results: P1={w1}, P2={w2}, D={d}")
    return per_game


def time_training_forward(num_samples=10_000, test_epochs=3):
    """
    Roughly estimate time for one FULL training epoch on the replay buffer.
    We only time forward passes on a dummy dataset, then double it to
    approximate forward+backward cost.
    """
    print(f"\nTiming training forward pass on {num_samples} samples...")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(num_samples, INPUT_SIZE)).astype(np.float32)

    # Dummy network weights with same sizes as your SimpleNN(hidden=256)
    W1 = rng.normal(scale=0.1, size=(HIDDEN_SIZE, INPUT_SIZE)).astype(np.float32)
    b1 = np.zeros((HIDDEN_SIZE,), dtype=np.float32)
    W2 = rng.normal(scale=0.1, size=(HIDDEN_SIZE,)).astype(np.float32)
    b2 = np.float32(0.0)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    t0 = time.perf_counter()
    for _ in range(test_epochs):
        z1 = X @ W1.T + b1            # (N, H)
        a1 = np.maximum(z1, 0.0)      # ReLU
        z2 = a1 @ W2 + b2             # (N,)
        _y = sigmoid(z2)              # predictions (we ignore loss)
    t1 = time.perf_counter()

    forward_total = t1 - t0
    forward_per_epoch = forward_total / float(test_epochs)

    # backprop is roughly another forward in cost; fudge factor 2x
    full_epoch_estimate = 2.0 * forward_per_epoch

    print(f"  Forward-only: {forward_total:.3f}s for {test_epochs} epochs "
          f"-> {forward_per_epoch:.3f}s / epoch")
    print(f"  Approx forward+backward: {full_epoch_estimate:.3f}s / epoch "
          f"on {num_samples} samples")

    return full_epoch_estimate, num_samples


def time_evaluation(num_models_sample=5, games_vs_prev_sample=20, games_vs_random_sample=40):
    """
    Benchmark evaluate_models on a small set of dummy models, then
    scale to your full (ITERATIONS+1) models and the big game counts.
    """
    print(f"\nTiming evaluation with {num_models_sample} dummy models...")
    models = [SimpleNN(hidden=HIDDEN_SIZE) for _ in range(num_models_sample)]

    # Theoretical #games used by evaluate_models with these params:
    #  (n-1)*games_vs_prev + games_vs_random
    small_total_games = (num_models_sample - 1) * games_vs_prev_sample + games_vs_random_sample

    t0 = time.perf_counter()
    evaluate_models(
        models,
        games_vs_prev=games_vs_prev_sample,
        games_vs_random=games_vs_random_sample,
    )
    t1 = time.perf_counter()

    total = t1 - t0
    per_game = total / float(small_total_games)
    print(f"  Took {total:.3f}s total for ~{small_total_games} games "
          f"-> {per_game:.4f}s / game (evaluation)")

    return per_game


def main():
    # 1) Benchmark small samples
    rand_pg = time_random_games(200)
    mcts_pg = time_mcts_games(200)
    train_epoch_time_sample, train_n_sample = time_training_forward()

    eval_per_game = time_evaluation(
        num_models_sample=5,
        games_vs_prev_sample=20,
        games_vs_random_sample=40,
    )

    # 2) Scale up to your full self-play workload
    total_rand_games = GAMES_INITIAL
    total_mcts_games = ITERATIONS * GAMES_PER_ITER  # training self-play

    est_rand_time = rand_pg * total_rand_games
    est_mcts_time = mcts_pg * total_mcts_games
    est_selfplay_seconds = est_rand_time + est_mcts_time

    # 3) Scale training time to full replay buffer & all iterations
    # Assume replay buffer quickly reaches REPLAY_MAX_SIZE and stays near that.
    scale_factor = REPLAY_MAX_SIZE / float(train_n_sample)
    est_train_epoch_full = train_epoch_time_sample * scale_factor
    total_train_epochs = ITERATIONS * TRAIN_EPOCHS
    est_training_seconds = est_train_epoch_full * total_train_epochs

    # 4) Scale evaluation time to your real params
    num_models_full = ITERATIONS + 1
    games_vs_prev_full = 1000
    games_vs_random_full = 4000
    total_eval_games_full = (num_models_full - 1) * games_vs_prev_full + games_vs_random_full
    est_eval_seconds = eval_per_game * total_eval_games_full

    # 5) Summaries
    print("\n========== ESTIMATED RUNTIME BREAKDOWN ==========")
    print(f"Warmup random self-play: {est_rand_time/60:.1f} minutes")
    print(f"Training self-play (MCTS): {est_mcts_time/60:.1f} minutes")
    print(f"  -> Total self-play: {est_selfplay_seconds/3600:.2f} hours")

    print(f"\nApprox NN training time:")
    print(f"  Per epoch on ~{REPLAY_MAX_SIZE} samples: {est_train_epoch_full:.2f}s")
    print(f"  Total epochs ({total_train_epochs}): {est_training_seconds/3600:.2f} hours")

    print(f"\nApprox final evaluation time:")
    print(f"  Total eval games: {total_eval_games_full} "
          f"(prev: {games_vs_prev_full}, random: {games_vs_random_full})")
    print(f"  Eval time: {est_eval_seconds/3600:.2f} hours")

    total_seconds = est_selfplay_seconds + est_training_seconds + est_eval_seconds
    print("\n========== ESTIMATED TOTAL RUNTIME ==========")
    print(f"Estimated total time for full train.py: {total_seconds/3600:.2f} hours")
    print("\n(These are rough estimates; real time may differ, "
          "but this should give you the right ballpark.)")


if __name__ == "__main__":
    main()
