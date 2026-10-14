"""Evaluation-only timing/target checks; never used by model or controller."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument("root",type=Path);args=p.parse_args();root=args.root
summary=json.loads((root/"summary.json").read_text());detail=[]
for r in summary:
    directory=root/f"{r['scenario']}_seed{r['seed']}_{r['mode']}"
    apps=[json.loads(x) for x in (directory/"motor_applications.jsonl").read_text().splitlines()]
    brains=[json.loads(x) for x in (directory/"brain_applications.jsonl").read_text().splitlines()]
    for e in r["reactions"]:
        selected=[a for a in apps if a.get("motor_selected") and a.get("motor_changed") and a.get("action") in e["expected"] and a["observed_sim_s"]>=e["onset_sim_s"]]
        first=min(selected,key=lambda a:a["applied_sim_s"]) if selected else None
        row={"mode":r["mode"],"scenario":r["scenario"],"seed":r["seed"],"kind":e["kind"],"matched_within_window":bool(e["first_action"]),"first_changed_after_onset_sim_s":first["applied_sim_s"]-e["onset_sim_s"] if first else None,"first_changed_source":first.get("source") if first else None,"late_after_event_window":bool(first and first["applied_sim_s"]>=e["end_sim_s"]),"first_correct_target_plan_sim_s":None,"correct_target_plan_error_m":None}
        if e["kind"]=="target":
            for b in sorted(brains,key=lambda a:a["applied_sim_s"]):
                if not b["accepted"] or b["observed_sim_s"]<e["onset_sim_s"] or "motor_plan" not in b:continue
                progress=min(1,(b["applied_sim_s"]-e["onset_sim_s"])/.7)
                goal=[r["scenario_parameters"]["nominal_goal"][0],e["start_y"]+(e["end_y"]-e["start_y"])*progress]
                error=float(np.linalg.norm(np.asarray(b["motor_plan"]["calibrated_goal_m"][:2])-goal))
                if error<.04:
                    row["first_correct_target_plan_sim_s"]=b["applied_sim_s"]-e["onset_sim_s"]
                    row["correct_target_plan_error_m"]=error
                    break
        detail.append(row)
groups={}
for mode in ("vlm","vlm_omni"):
    cohort=[e for e in detail if e["mode"]==mode];episodes=[r for r in summary if r["mode"]==mode]
    avoid=[e for e in cohort if e["kind"]!="target"];targets=[e for e in cohort if e["kind"]=="target"]
    groups[mode]={"avoidance_matched":sum(e["matched_within_window"] for e in avoid),"avoidance_events":len(avoid),"target_motor_matched":sum(e["matched_within_window"] for e in targets),"target_events":len(targets),"correct_target_plans":sum(e["first_correct_target_plan_sim_s"] is not None for e in targets),"correct_target_plan_latency_sim_s":[e["first_correct_target_plan_sim_s"] for e in targets if e["first_correct_target_plan_sim_s"] is not None],"late_matched_commands":sum(e["late_after_event_window"] for e in cohort),"quiet_intervention_ticks":sum(r["clean_prefix_model_intervention_ticks"] for r in episodes if r["scenario"]=="quiet"),"quiet_ticks":sum(r["control_steps"] for r in episodes if r["scenario"]=="quiet")}
result={"groups":groups,"events":detail,"scope":"Category-matched new actions and accurate goal-plan updates are separate; neither is hazard recall","analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/"motion_metrics.json").write_text(json.dumps(result,indent=2));print(json.dumps(groups,indent=2))
