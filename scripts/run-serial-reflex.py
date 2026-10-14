"""Freeze source and run serial paired brain/cerebellum experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[541,577,613])
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--modes", nargs=2, choices=["omni","vlm","vlm_omni"], default=["vlm","vlm_omni"])
parser.add_argument("--scenarios", nargs="+", choices=["quiet","hand","slider","target","occlusion","combined","hand_close","slider_fast","occlusion_brief","target_jump"], default=["quiet","hand_close","slider_fast","occlusion_brief","target_jump"])
parser.add_argument("--freeze-bank", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
if args.out.exists() or len(set(args.modes)) != 2:
    raise SystemExit("Use a fresh output directory and two distinct modes")
args.out.mkdir(parents=True)
source = Path(__file__).resolve().parents[1]/"jev_reflex"
shutil.copytree(source, args.out/"source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2(__file__, args.out/"driver_source.py")
manifest = {"protocol":"brain_cerebellum_comparison_v1",
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
            "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "arguments":{"seeds":args.seeds,"seconds":args.seconds,"scenarios":args.scenarios,"modes":args.modes,"freeze_bank":args.freeze_bank},
            "execution":"serial physics episodes, same continuous visual-brain request strategy, mode order reversed by seed", "commands":[]}
summaries=[]
for index,seed in enumerate(args.seeds):
    modes=args.modes if index%2==0 else list(reversed(args.modes))
    output=args.out/f"seed{seed}"
    command=[sys.executable,"-u","-m","jev_reflex.run","--out",str(output),"--scenarios",*args.scenarios,"--seeds",str(seed),"--modes",*modes,"--seconds",str(args.seconds),"--image-size","320","--brain-interval","0"]
    if args.freeze_bank:command.append("--freeze-bank")
    manifest["commands"].append(command)
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"starting_seed":seed,"modes":modes,"scenarios":args.scenarios}),flush=True)
    completed=subprocess.run(command)
    if completed.returncode:raise SystemExit(f"Seed {seed} runner failed: {completed.returncode}")
    summaries.extend(json.loads((output/"summary.json").read_text()))
    for episode in output.iterdir():
        if episode.is_dir():shutil.move(str(episode),args.out/episode.name)
    (args.out/"summary.json").write_text(json.dumps(summaries,indent=2))
print(json.dumps({"episodes":len(summaries),"output":str(args.out)}),flush=True)
