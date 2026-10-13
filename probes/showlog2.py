import json, sys, collections
d = json.load(open(sys.argv[1]))
print("success", d["success"], "decisions", d["decisions"], "sim_s", d["sim_seconds"], "wall", d.get("wall_s"))
print(collections.Counter(r["phase"] for r in d["log"]))
# show phase transitions
prev = None
for r in d["log"]:
    if r["phase"] != prev:
        print(f"d{r['d']:3d} t={r['t']:6.1f} -> {r['phase']:10s} eef={r['eef']} hold={r['holding']} prog={r['progress']}")
        prev = r["phase"]
