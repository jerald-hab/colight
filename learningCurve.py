import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# CHANGE THIS to the experiment you want to plot
base_path = "records/0404_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_04_05_09_42_56/train_round"
round_numbers = []
avg_queues = []

round_dirs = sorted(
    [d for d in os.listdir(base_path) if d.startswith("round_")],
    key=lambda x: int(x.split("_")[1])
)

for round_name in round_dirs:
    round_idx = int(round_name.split("_")[1])
    round_path = os.path.join(base_path, round_name)

    total_queue_all_intersections = []

    print(f"Processing {round_name}...")

    for generator in os.listdir(round_path):
        gen_path = os.path.join(round_path, generator)

        for f in os.listdir(gen_path):
            if f.startswith("inter_") and f.endswith(".pkl"):
                full_path = os.path.join(gen_path, f)

                if os.path.getsize(full_path) == 0:
                    print("  Skipping empty file:", full_path)
                    continue

                try:
                    with open(full_path, "rb") as file:
                        data = pickle.load(file)
                except EOFError:
                    print("  Skipping corrupted file:", full_path)
                    continue

                rewards = [step["reward"] for step in data]
                total_queue_all_intersections.append(np.mean(rewards))

    if total_queue_all_intersections:
        round_avg = np.mean(total_queue_all_intersections)
        round_numbers.append(round_idx)
        avg_queues.append(round_avg)
        print(f"  Round {round_idx}: Avg Reward = {round_avg:.2f}")
    else:
        print(f"  No valid data for {round_name}")

print("Final round_numbers:", round_numbers)
print("Final avg_queues:", avg_queues)

plt.figure(figsize=(8,5))
plt.plot(round_numbers, avg_queues, marker='o')
plt.xlabel("Training Round")
plt.ylabel("Average Reward")
plt.title("6x6 Learning Curve: Average Reward vs. Training Round")
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve_queue6.png", dpi=150)
plt.show()

