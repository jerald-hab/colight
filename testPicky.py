import pickle

# Update this path to any pickle you want to inspect
pkl_path = "records/0515_afternoon_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_04_01_16_14_31/train_round/round_9/generator_1/inter_18.pkl"

print("Loading:", pkl_path)

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Length:", len(data))

if len(data) > 0:
    print("First entry keys:", data[0].keys())
    print("First entry:", data[0])
else:
    print("The pickle file is EMPTY.")
