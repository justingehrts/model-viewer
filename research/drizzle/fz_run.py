import sys; sys.path.insert(0, "."); import json
from fetch_history import FAM, get
k = int(sys.argv[1]); tasks = json.load(open("fz_tasks.json"))[k::3]
for s, lat, lon, a, b, n in tasks:
    for fam, v in FAM.items(): get(lat, lon, a, b, v, f"{s}_{a}_{fam}")
    print(s, a, "done", flush=True)
print("DONE", flush=True)
