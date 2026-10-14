"""Replay deterministic arbitration against every recorded control tick."""
import argparse,hashlib,json,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_reflex.control import choose_motor
from jev_reflex.policy import ActionMailbox

def rows(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def median(values):return statistics.median(values) if values else None
p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args();root=a.root
manifest=json.loads((root/"manifest.json").read_text())
for name in ("control.py","policy.py"):
    assert hashlib.sha256((Path(__file__).resolve().parents[1]/"jev_reflex"/name).read_bytes()).hexdigest()==manifest["source_sha256"][name]
summary=json.loads((root/"summary.json").read_text());groups={};all_events=[];all_selected=[];checks=[]
for result in summary:
    directory=root/f"{result['scenario']}_seed{result['seed']}_{result['mode']}"
    brain=defaultdict(list);local=defaultdict(list)
    for record in rows(directory/"brain_applications.jsonl"):brain[round(record["applied_sim_s"]/.05)].append(record)
    for record in rows(directory/"applications.jsonl"):local[round(record["applied_sim_s"]/.05)].append(record)
    limits=result["software_limits"];cerebellum=ActionMailbox(limits["max_observation_age_s"],limits["command_ttl_s"])
    cortex=ActionMailbox(limits["max_brain_age_s"],limits["brain_command_ttl_s"])
    previous_identity=None;previous_action="continue";previous_velocity=np.zeros(3);selected=[];mismatches=[]
    states=rows(directory/"states.jsonl")
    for state in states:
        tick=round(state["sim_s"]/.05)-1
        for record in brain[tick]:
            if record["accepted"]:
                if record.get("motor_plan",{}).get("changed"):cerebellum.record=None
                cortex.record=record
        for record in local[tick]:
            if record["accepted"]:cerebellum.record=record
        action,scale,source,record=choose_motor(state["wall_s"],cerebellum,cortex,result["mode"]=="vlm_omni",True)
        if action!=state["action"] or source!=state["motor_source"] or abs(scale-state["model_speed_scale"])>1e-8:
            mismatches.append({"tick":tick,"predicted":[action,source,scale],"recorded":[state["action"],state["motor_source"],state["model_speed_scale"]]})
        identity=(source,record["sequence"]) if record else (source,None)
        if record and identity!=previous_identity:
            changed=action!=previous_action and (np.linalg.norm(np.asarray(state["applied_velocity"])-previous_velocity)>.005 or state["plan_offset_changed"])
            selected.append({"source":source,"sequence":record["sequence"],"action":action,"observed_sim_s":record["observed_sim_s"],"observed_wall_s":record["observed_wall_s"],"selected_sim_s":state["sim_s"]-.05,"selected_wall_s":state["wall_s"],"observation_age_s":state["wall_s"]-record["observed_wall_s"],"motor_changed":bool(changed),"cached_activation":state["wall_s"]-record["applied_wall_s"]>.001,"gate_triggered":record.get("model_intervention"),"mode":result["mode"],"scenario":result["scenario"],"seed":result["seed"]})
        previous_identity,previous_action,previous_velocity=identity,action,np.asarray(state["applied_velocity"])
    checks.append({"episode":directory.name,"ticks":len(states),"mismatch_count":len(mismatches),"mismatches":mismatches[:10]})
    for event in result["reactions"]:
        candidates=[s for s in selected if s["motor_changed"] and s["action"] in event["expected"] and s["observed_sim_s"]>=event["onset_sim_s"] and s["selected_sim_s"]<event["end_sim_s"]]
        first=candidates[0] if candidates else None
        all_events.append({"mode":result["mode"],"scenario":result["scenario"],"seed":result["seed"],"kind":event["kind"],"onset_sim_s":event["onset_sim_s"],"first":first,"onset_to_selection_sim_s":first["selected_sim_s"]-event["onset_sim_s"] if first else None})
    all_selected.extend(selected)
assert not any(c["mismatch_count"] for c in checks),json.dumps([c for c in checks if c["mismatch_count"]])
for mode in ("vlm","vlm_omni"):
    selected=[s for s in all_selected if s["mode"]==mode];changed=[s for s in selected if s["motor_changed"]];events=[e for e in all_events if e["mode"]==mode]
    groups[mode]={"selected_records":len(selected),"cached_activations":sum(s["cached_activation"] for s in selected),"changed_records":len(changed),"selected_capture_to_motor_p50_s":median([s["observation_age_s"] for s in selected]),"changed_capture_to_motor_p50_s":median([s["observation_age_s"] for s in changed]),"matched_events":sum(e["first"] is not None for e in events),"matched_avoidance_events":sum(e["first"] is not None for e in events if e["kind"]!="target"),"events":len(events),"sources":dict(Counter(s["source"] for s in selected)),"triggered_local_changed_records":sum(s["source"]=="cerebellum" and s["gate_triggered"] and s["action"]!="continue" for s in changed)}
output={"scope":"actual selection changes including cached activations; deterministic replay matches every recorded tick","groups":groups,"events":all_events,"checks":checks,"selected":all_selected,"analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root/"selection_metrics.json").write_text(json.dumps(output,indent=2));print(json.dumps({"groups":groups,"ticks_verified":sum(c["ticks"] for c in checks),"matched_events":[e for e in all_events if e["first"]]},indent=2))
