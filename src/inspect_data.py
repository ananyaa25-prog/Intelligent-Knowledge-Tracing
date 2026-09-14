# src/inspect_data.py — one-off inspection script, not part of the pipeline

import pickle

with open("data/processed/windowed_data.pkl", "rb") as f:
    data = pickle.load(f)

print("Top-level keys:", list(data.keys()))
print("splits keys:", list(data["splits"].keys()))

train = data["splits"]["train"]
print("train type:", type(train))

if isinstance(train, dict):
    print("train inner keys:", list(train.keys()))
    for k, v in train.items():
        try:
            shape = getattr(v, "shape", None)
            if shape is not None:
                print(f"  {k}: shape = {shape}, dtype = {v.dtype}")
            else:
                print(f"  {k}: len = {len(v)}, type = {type(v)}")
        except Exception as e:
            print(f"  {k}: could not inspect ({e})")

elif isinstance(train, list):
    print("train length:", len(train))
    first = train[0]
    print("first element type:", type(first))
    if hasattr(first, "keys"):
        print("first element keys:", list(first.keys()))
    else:
        print("first element:", first)

else:
    print("Unexpected train type — dumping repr:")
    print(repr(train)[:500])
print("\n--- skill_graph.pkl ---")
with open("data/processed/skill_graph.pkl", "rb") as f:
    graph_data = pickle.load(f)
print("skill_graph keys:", list(graph_data.keys()))
for k, v in graph_data.items():
    try:
        shape = getattr(v, "shape", None)
        print(f"  {k}: shape = {shape}" if shape is not None else f"  {k}: type = {type(v)}")
    except Exception as e:
        print(f"  {k}: could not inspect ({e})")