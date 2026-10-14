"""Freeze all eligible exact cases and provenance before any paired outcomes."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,nargs="+",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
qs=[json.loads((d/"qualification.json").read_text()) for d in a.inputs]
for q in qs:
    for field in ["source_sha256","model_policy","seconds"]:assert q[field]==qs[0][field],field
cases=[k for q in qs for k in q["outcomes"]];assert len(cases)==len(set(cases)),"Duplicate screening cases"
driver=hashlib.sha256((Path(__file__).resolve().parent/"run-qualified-comparison.py").read_bytes()).hexdigest()
assert all(json.loads((d/"manifest.json").read_text())["driver_sha256"]==driver for d in a.inputs)
merged={**qs[0],"qualified_cases":[k for q in qs for k in q["qualified_cases"]],"outcomes":{k:v for q in qs for k,v in q["outcomes"].items()},
        "screening_sources":[str(d) for d in a.inputs],"qualification_sha256":[hashlib.sha256((d/"qualification.json").read_bytes()).hexdigest() for d in a.inputs],"paired_repeats":3,"driver_sha256":driver}
with a.out.open("x") as f:json.dump(merged,f,indent=2)
print(json.dumps({"qualified_cases":merged["qualified_cases"],"outcomes":merged["outcomes"]}))
