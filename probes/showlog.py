import json, sys
d = json.load(open(sys.argv[1]))
rows = d["log"]
sel = rows[:5] + rows[len(rows)//2:len(rows)//2+3] + rows[-5:]
for r in sel:
    print(r["d"], r["phase"], "eef", r["eef"], "ee_px", r.get("ee_px"),
          "tgt_px", r.get("tgt_px"), "px_err", r.get("px_err"), "vel", r.get("vel"))
