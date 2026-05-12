import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import datetime

# CHANGE THIS to the experiment you want to plot
uni_csv = "records/0428_Colight_6_6_uni/anon_6_6_300_0.3_uni.json_04_28_05_30_50/running_time.csv"
bi_csv  = "records/0502_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_05_02_10_29_34/running_time.csv"
ten_csv = "records/0410_Colight_10_10_bi/anon_10_10_300_0.3_bi.json_04_10_09_59_24/running_time.csv"

df_uni = pd.read_csv(uni_csv, delim_whitespace=True)
df_bi  = pd.read_csv(bi_csv, delim_whitespace=True)
df_10  = pd.read_csv(ten_csv, delim_whitespace=True)

# Extract per-round total time
t_uni = df_uni["all_times"].tolist()
t_bi  = df_bi["all_times"].tolist()
t_10  = df_10["all_times"].tolist()

# Episode numbers
rounds_uni = list(range(len(t_uni)))
rounds_bi  = list(range(len(t_bi)))
rounds_10  = list(range(len(t_10)))

plt.figure(figsize=(10,6))

plt.plot(rounds_uni, t_uni, color="gold", label="6×6 Uni", linewidth=2)
plt.plot(rounds_bi,  t_bi,  color="red",  label="6×6 Bi", linewidth=2)
plt.plot(rounds_10,  t_10,  color="green", label="10×10 Bi", linewidth=2)

plt.xlabel("Training Round")
plt.ylabel("Time per Round (seconds)")
plt.title("Training Time per Round Across Grid Sizes")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("training_time_per_round_comparison.png", dpi=150)
plt.show()
