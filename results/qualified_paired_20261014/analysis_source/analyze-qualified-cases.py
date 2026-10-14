"""Analyze frozen repeated cases; failures remain in denominators and paired speed needs two successes."""
import argparse,hashlib,json,statistics
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
rows=json.loads((a.root/"summary.json").read_text());manifest=json.loads((a.root/"manifest.json").read_text())
if manifest["phase"]=="paired":
    assert manifest["driver_sha256"]==manifest["qualification"]["driver_sha256"], "Driver changed after screening"
    assert manifest["repeats"]==manifest["qualification"]["paired_repeats"]==3
    assert manifest["source_sha256"]==manifest["qualification"]["source_sha256"]
    assert manifest["model_policy"]==manifest["qualification"]["model_policy"]
by_case=defaultdict(list)
for row in rows:by_case[row["case_id"]].append(row)
result={"phase":manifest["phase"],"episodes":len(rows),"cases":{},"pairs":[]}
for case,items in by_case.items():
    result["cases"][case]={}
    for mode in sorted({r["mode"] for r in items}):
        group=[r for r in items if r["mode"]==mode]
        good=[r for r in group if r["task_completion"]["success"]]
        result["cases"][case][mode]={"successes":len(good),"runs":len(group),
            "completion_median_wall_s":statistics.median(r["task_completion"]["retained_completion_wall_s"] for r in good) if good else None,
            "recovery_median_wall_s":statistics.median(r["task_completion"]["recovery_wall_s"] for r in good) if good else None,
            "first_dwell_median_wall_s":statistics.median(r["task_completion"]["first_dwell_wall_s"] for r in group if r["task_completion"]["first_dwell_wall_s"] is not None) if any(r["task_completion"]["first_dwell_wall_s"] is not None for r in group) else None,
            "watchdog_fraction":sum(r["watchdog_ticks"] for r in group)/sum(r["control_steps"] for r in group),
            "brain_errors":sum(r["brain_errors"] for r in group),"cerebellum_errors":sum(r["api_errors"] for r in group),
            "human_contact_peak_N":max(r["peak_human_contact_N"] for r in group),"object_contact_peak_N":max(r["peak_object_contact_N"] for r in group)}
    if manifest["phase"]=="paired":
        for repeat in sorted({r["repeat"] for r in items}):
            pair={r["mode"]:r for r in items if r["repeat"]==repeat}
            assert set(pair)=={"vlm","vlm_omni"}
            b,c=pair["vlm"],pair["vlm_omni"]
            for key in ["initial_state_sha256","scenario_parameters","nominal_motor_plan","software_limits"]:assert b[key]==c[key],key
            success=b["task_completion"]["success"] and c["task_completion"]["success"]
            delta=b["task_completion"]["retained_completion_wall_s"]-c["task_completion"]["retained_completion_wall_s"] if success else None
            result["pairs"].append({"case_id":case,"repeat":repeat,"brain_success":b["task_completion"]["success"],"mixed_success":c["task_completion"]["success"],"mixed_seconds_saved":delta,"brain_dir":b["episode_dir"],"mixed_dir":c["episode_dir"]})
result["speed_scope"]="Positive paired seconds saved means faster final retained task completion; only pairs with both successes have a speed difference. All failures stay in case counts."
result["analysis_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(a.root/"task_comparison.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
