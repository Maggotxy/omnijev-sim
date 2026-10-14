"""Screen brain-only competence, then freeze and compare identical qualified cases."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_reflex.run import episode
from jev_reflex.scenario import Scenario

p=argparse.ArgumentParser()
p.add_argument("--out",type=Path,required=True)
p.add_argument("--phase",choices=["screen","paired"],required=True)
p.add_argument("--cases",nargs="+",default=["quiet:701","occlusion_brief:709","target_jump:719"])
p.add_argument("--repeats",type=int,default=3)
p.add_argument("--seconds",type=float,default=36)
p.add_argument("--qualification",type=Path)
a=p.parse_args()
if a.out.exists():raise SystemExit("Use a fresh output directory")
source=Path(__file__).resolve().parents[1]/"jev_reflex"
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
qualified=None
if a.phase=="paired":
    if not a.qualification:raise SystemExit("Paired run requires a prior frozen qualification")
    qualified=json.loads(a.qualification.read_text())
    if hashes!=qualified["source_sha256"]:raise SystemExit("Runtime changed after qualification")
    if os.environ.get("GEMINI_MODEL","gemini-3.8-flash")!=qualified["model_policy"]["brain"] or os.environ.get("OMNIJEV_MODEL","omnijev-0.8b-kk")!=qualified["model_policy"]["cerebellum"]:
        raise SystemExit("Model identifiers changed after qualification")
    a.cases=qualified["qualified_cases"]
    a.seconds=qualified["seconds"]
    if not a.cases:raise SystemExit("No qualified brain-only cases")
a.out.mkdir(parents=True)
shutil.copytree(source,a.out/"source_snapshot",ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2(__file__,a.out/"driver_source.py")
manifest={"protocol":"qualified_brain_cases_v1","phase":a.phase,"cases":a.cases,"repeats":a.repeats,"seconds":a.seconds,"source_sha256":hashes,"driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "model_policy":{"brain":os.environ.get("GEMINI_MODEL","gemini-3.8-flash"),"cerebellum":os.environ.get("OMNIJEV_MODEL","omnijev-0.8b-kk"),"weights":"fixed","brain_interval":0,"frozen_bank":True},
          "selection_rule":"Each exact case must pass all 3 fresh brain-only screening repetitions; mixed outcomes never affect selection", "qualification":qualified}
(a.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
args=argparse.Namespace(asset_root=source.parent/"resources/panda",image_size=320,api_timeout=2,max_observation_age=1.2,command_ttl=1.4,seconds=a.seconds,render_hz=8,brain_timeout=15,brain_interval=0,max_brain_age=8,brain_command_ttl=10,freeze_bank=True)
rows=[]
for repeat in range(a.repeats):
    for index,case in enumerate(a.cases):
        kind,seed=case.split(":");scenario=Scenario.generate(kind,int(seed))
        modes=["vlm"] if a.phase=="screen" else (["vlm","vlm_omni"] if (repeat+index)%2==0 else ["vlm_omni","vlm"])
        for mode in modes:
            out=a.out/f"repeat{repeat+1}"/f"{kind}_seed{seed}_{mode}"
            print(json.dumps({"phase":a.phase,"repeat":repeat+1,"case":case,"mode":mode}),flush=True)
            row=episode(args,scenario,mode,out)
            row.update(case_id=case,repeat=repeat+1,episode_dir=str(out.relative_to(a.out)))
            rows.append(row);(a.out/"summary.json").write_text(json.dumps(rows,indent=2))
            print(json.dumps({"completed":len(rows),"case":case,"mode":mode,"task":row["task_completion"]}),flush=True)
if a.phase=="screen":
    cases={c:[r["task_completion"]["success"] for r in rows if r["case_id"]==c] for c in a.cases}
    receipt={"qualified_cases":[c for c,outcomes in cases.items() if len(outcomes)>=3 and all(outcomes)],"outcomes":cases,"seconds":a.seconds,"source_sha256":hashes,"model_policy":manifest["model_policy"],"rule":manifest["selection_rule"]}
    (a.out/"qualification.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps({"finished":a.phase,"episodes":len(rows)}),flush=True)
