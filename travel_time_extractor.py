import pandas as pd
import glob
import os

# Path to the round folder (NOT generator folder) 

#base = "/home/jerald/colight/records/0404_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_04_08_18_24_35/train_round"
base = "/home/jerald/colight/records/0501_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_05_01_20_31_06/train_round"
# Find the highest-numbered round folder
rounds = [d for d in os.listdir(base) if d.startswith("round_")]
round_nums = sorted([int(r.split("_")[1]) for r in rounds])
final_round = f"round_{round_nums[-1]}"

path = os.path.join(base, final_round)
print("Using final round:", path)

# Find all vehicle_inter_*.csv files inside all generator folders
files = glob.glob(os.path.join(path, "generator_*", "vehicle_inter_*.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    
    # Use travel_time if present, otherwise compute it
    if "travel_time" in df.columns:
        df["tt"] = df["travel_time"]
    else:
        df["tt"] = df["leave_time"] - df["enter_time"]
    
    dfs.append(df)

# Combine all intersections from all generators
all_data = pd.concat(dfs, ignore_index=True)

# Compute metrics
avg_tt = all_data["tt"].mean()
median_tt = all_data["tt"].median()
count = len(all_data)

print("Total vehicles:", count)
print("Average travel time:", avg_tt)
print("Median travel time:", median_tt)
