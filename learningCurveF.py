import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_round_rewards(round_dir):
    """Load all inter_*.pkl files inside a round directory and average their rewards."""
    rewards = []

    for fname in os.listdir(round_dir):
        if fname.startswith("inter_") and fname.endswith(".pkl"):
            fpath = os.path.join(round_dir, fname)
            with open(fpath, "rb") as f:
                data = pickle.load(f)

            # Extract reward from each step
            step_rewards = [step["reward"] for step in data]
            rewards.append(np.mean(step_rewards))

    if len(rewards) == 0:
        return None

    return float(np.mean(rewards))


def generate_learning_curve(train_round_dir):
    """train_round_dir contains round_0, round_1, round_2, ..."""
    rounds = []
    avg_rewards = []

    for name in sorted(os.listdir(train_round_dir)):
        if name.startswith("round_"):
            round_idx = int(name.split("_")[1])
            round_path = os.path.join(train_round_dir, name)

            avg_reward = load_round_rewards(round_path)
            if avg_reward is not None:
                rounds.append(round_idx)
                avg_rewards.append(avg_reward)
                print(f"Round {round_idx}: {avg_reward:.4f}")
            else:
                print(f"Round {round_idx}: no data")

    return rounds, avg_rewards


def plot_curve(rounds, rewards, out_path="learning_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, rewards, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Average Reward")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing round_x folders")
    parser.add_argument("--out", type=str, default="learning_curve.png",
                        help="Output plot filename")
    args = parser.parse_args()

    rounds, rewards = generate_learning_curve(args.dir)
    plot_curve(rounds, rewards, args.out)
