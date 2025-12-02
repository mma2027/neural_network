import os
import numpy as np
from neural_network import SimpleNN
from helper import evaluate_models, color, CYAN, BOLD
import csv

MODELS_DIR = "models"


def load_all_models():
    files = sorted(os.listdir(MODELS_DIR))

    # Prioritize proper numeric order:
    def sort_key(name):
        # model_gen_XX.npz → number
        if name.startswith("model_gen_") and name.endswith(".npz"):
            try:
                return int(name[len("model_gen_"):-4])
            except:
                return 10**9

        # model_final.npz → put at the END
        if name == "model_final.npz":
            return 10**12

        # Any other file → far end
        return 10**11

    files = sorted(files, key=sort_key)

    models = []
    for fname in files:
        if not fname.endswith(".npz"):
            continue

        path = os.path.join(MODELS_DIR, fname)
        print(color(f"Loading: {path}", CYAN, BOLD))
        try:
            model = SimpleNN.load(path)
            models.append(model)
        except Exception as e:
            print(color(f"  Failed to load {fname}: {e}", CYAN))
            continue

    return models


def write_csv(rows, path="model_eval_summary.csv"):
    with open(path, "w", newline="") as f:
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
    print(color(f"\nCSV saved to {path}\n", CYAN, BOLD))


def main():
    print(color("\n===============================================", BOLD, CYAN))
    print(color("       Loading ALL models for evaluation       ", BOLD, CYAN))
    print(color("===============================================\n", BOLD, CYAN))

    models = load_all_models()

    if len(models) < 2:
        print(color("Not enough models to evaluate!", CYAN, BOLD))
        return

    print(color("\nEvaluating model ladder...\n", BOLD, CYAN))

    rows = evaluate_models(
        models,
        games_vs_prev=200,
        games_vs_random=400
    )

    write_csv(rows)


if __name__ == "__main__":
    main()
