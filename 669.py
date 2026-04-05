import os, pickle
import numpy as np
import pickle, os

base_path = "records/0404_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_04_04_19_28_05/train_round"
for round_name in sorted(os.listdir(base_path)):
    round_path = os.path.join(base_path, round_name)
    if not round_name.startswith("round_"):
        continue

    total_queue_all_intersections = []

    for generator in os.listdir(round_path):
        gen_path = os.path.join(round_path, generator)

        for f in os.listdir(gen_path):
            if f.startswith("inter_") and f.endswith(".pkl"):
                with open(os.path.join(gen_path, f), "rb") as file:
                    data = pickle.load(file)

                    # compute total queue per timestep
                    queues = [
                        sum(step["state"]["lane_num_vehicle"])
                        for step in data
                    ]

                    total_queue_all_intersections.append(np.mean(queues))

    print(round_name, "Average Queue:", np.mean(total_queue_all_intersections))
