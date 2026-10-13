"""Freeze runtime source and run pairs serially, balancing the mode order by seed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[233, 257])
parser.add_argument("--seconds", type=float, default=40)
parser.add_argument("--scenarios", nargs="+", choices=["quiet", "hand", "slider", "target", "occlusion", "combined"], default=["quiet", "hand", "slider", "target", "occlusion", "combined"])
args = parser.parse_args()
if args.out.exists():
    raise SystemExit("Use a fresh output directory")
args.out.mkdir(parents=True)
source = Path(__file__).resolve().parents[1]/"jev_reflex"
shutil.copytree(source, args.out/"source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
manifest = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "arguments": {"seeds": args.seeds, "seconds": args.seconds, "scenarios": args.scenarios}, "execution": "serial, one physics episode and one reflex HTTP worker at a time",
            "commands": []}
summaries = []
for index, seed in enumerate(args.seeds):
    modes = ["omni", "vlm_omni"] if index % 2 == 0 else ["vlm_omni", "omni"]
    output = args.out/f"seed{seed}"
    command = [sys.executable, "-u", "-m", "jev_reflex.run", "--out", str(output), "--scenarios",
               *args.scenarios, "--seeds", str(seed),
               "--modes", *modes, "--seconds", str(args.seconds), "--image-size", "320"]
    manifest["commands"].append(command)
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    completed = subprocess.run(command)
    if completed.returncode:
        raise SystemExit(f"Seed {seed} runner failed: {completed.returncode}")
    summaries.extend(json.loads((output/"summary.json").read_text()))
    for episode in output.iterdir():
        if episode.is_dir():
            shutil.move(str(episode), args.out/episode.name)
    (args.out/"summary.json").write_text(json.dumps(summaries, indent=2))
print(json.dumps({"episodes": len(summaries), "output": str(args.out)}), flush=True)
