import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import datetime

# CHANGE THIS to the experiment you want to plot
base_path = "records/0410_Colight_10_10_bi/anon_10_10_300_0.3_bi.json_04_10_09_59_24/train_round"

round_numbers = []
avg_rewards = []
avg_travel_times = []
episode_timestamps = []

# Sort round folders numerically
round_dirs = sorted(
    [d for d in os.listdir(base_path) if d.startswith("round_")],
    key=lambda x: int(x.split("_")[1])
)

print(f"Found {len(round_dirs)} rounds.")

for round_name in round_dirs:
    round_idx = int(round_name.split("_")[1])
    round_path = os.path.join(base_path, round_name)

    reward_list = []
    travel_time_list = []

    print(f"\nProcessing {round_name}...")

    # -------------------------
    # Load timestamp if exists
    # -------------------------
    ts_file = os.path.join(round_path, "timestamp.txt")
    if os.path.exists(ts_file):
        try:
            with open(ts_file, "r") as f:
                ts = f.read().strip()
                episode_timestamps.append(datetime.fromisoformat(ts))
        except:
            print("  Could not parse timestamp:", ts_file)

    # -------------------------
    # Loop through generator folders
    # -------------------------
    for generator in os.listdir(round_path):
        gen_path = os.path.join(round_path, generator)

        if not os.path.isdir(gen_path):
            continue

        # -------------------------
        # Extract rewards
        # -------------------------
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
                reward_list.append(np.mean(rewards))

        # -------------------------
        # Extract travel time
        # -------------------------
        # print("Looking for CSV in:", gen_path)
        # print("Files:", os.listdir(gen_path))
        csv_files = glob.glob(os.path.join(gen_path, "vehicle_inter_*.csv"))

        for csv in csv_files:
            df = pd.read_csv(csv)

            # Fix unnamed first column
            if df.columns[0].startswith("Unnamed"):
                df = df.rename(columns={df.columns[0]: "vehicle_id"})

            # Drop rows with missing times
            df = df.dropna(subset=["enter_time", "leave_time"])

            # Compute travel time
            df["tt"] = df["leave_time"] - df["enter_time"]

            # Only keep positive travel times
            df = df[df["tt"] >= 0]

            travel_time_list.extend(df["tt"].tolist())




    # Compute averages for this round
    avg_reward = np.mean(reward_list) if reward_list else None
    avg_tt = np.mean(travel_time_list) if travel_time_list else None

    avg_rewards.append(avg_reward)
    avg_travel_times.append(avg_tt)
    round_numbers.append(round_idx)

    print(f"  Round {round_idx}: Avg Reward = {avg_reward}, Avg Travel Time = {avg_tt}")


print("\nFinal round_numbers:", round_numbers)
print("Final avg_rewards:", avg_rewards)
print("Final avg_travel_times:", avg_travel_times)

# -----------------------------
# Plot Reward Curve
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(round_numbers, avg_rewards, marker='o')
plt.xlabel("Training Round")
plt.ylabel("Average Reward")
plt.title("Learning Curve: Average Reward vs. Training Round")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_reward_curve_10x10_bi.png", dpi=150)
print("Saved avg_reward_curve_6x6_uni.png")

# -----------------------------
# Plot Travel Time Curve
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(round_numbers, avg_travel_times, marker='o', color='orange')
plt.xlabel("Training Round")
plt.ylabel("Average Travel Time (s)")
plt.title("Learning Curve: Average Travel Time vs. Training Round")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_travel_time_curve_10x10_bi.png", dpi=150)
print("Saved avg_travel_ime_curve_6x6_uni.png")

# -----------------------------
# Compute Training Time
# -----------------------------
if len(episode_timestamps) > 1:
    total_seconds = (episode_timestamps[-1] - episode_timestamps[0]).total_seconds()
    total_minutes = total_seconds / 60
    print(f"\nTotal Training Time: {total_minutes:.2f} minutes")

    # Save to file so you can combine across experiments
    with open("training_time_minutes6x6_uni.txt", "w") as f:
        f.write(str(total_minutes))
else:
    print("\nNo timestamps found. Cannot compute training time.")

plt.show()