"""Evaluation-only task completion; truth never enters the motor policy."""
import math

def task_completion(states, scenario, dwell_s=2.0):
    events=scenario.events()
    reference=max([0.0]+[e["onset_sim_s"]+.7 if e["kind"]=="target" else e["end_sim_s"] for e in events])
    streak=0; first=None; retained=None; last=None; eligible=0
    reference_wall=0.0 if reference==0 else next((s.get("measured_wall_s",s["wall_s"]) for s in states if s["sim_s"]>=reference),reference)
    for row in states:
        speed=math.sqrt(sum(v*v for v in row["measured_eef_velocity_m_s"]))
        stable=row["sim_s"]>=reference and row["goal_reached"] and speed<.015
        if stable:
            eligible+=1
            streak+=1
            if streak*.05+1e-7>=dwell_s and retained is None:
                retained=row.get("measured_wall_s",row["wall_s"])
                if first is None:first=retained
        else:
            streak=0;retained=None
        last=row
    success=bool(last and streak*.05+1e-7>=dwell_s)
    return {"success":success,"criterion":"within 4cm XY / 5cm Z, speed < 0.015m/s continuously >=2s, retained through episode end after last event settles",
            "reference_sim_s":reference,"reference_wall_s":reference_wall,
            "first_dwell_wall_s":first,"retained_completion_wall_s":retained if success else None,
            "recovery_wall_s":max(0,retained-reference_wall) if success else None,
            "terminal_dwell_s":streak*.05,"stable_ticks_after_reference":eligible}
