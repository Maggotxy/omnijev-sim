"""Measured motion after matched commands; evaluation only, no policy access."""
import numpy as np


def motion_responses(scenario, applications, states):
    rows = []
    for event in scenario.events():
        if not states or event["onset_sim_s"] >= states[-1]["sim_s"]:
            continue
        candidates = [a for a in applications if a.get("motor_changed") and a.get("accepted")
                      and a.get("action") in event["expected"]
                      and a["observed_sim_s"] >= event["onset_sim_s"]
                      and a["applied_sim_s"] < event["end_sim_s"]]
        first = min(candidates, key=lambda a:a["applied_sim_s"]) if candidates else None
        row = {**event, "command_source": first.get("source") if first else None,
               "action": first["action"] if first else None, "command_sim_s": first["applied_sim_s"] if first else None,
               "command_to_motion_s": None, "onset_to_motion_s": None,
               "criterion": "5mm projected displacement, or speed halved below 0.015m/s for 3 ticks",
               "scope": "motion observed after a model command, not isolated causal attribution"}
        if first:
            before = next((s for s in reversed(states) if s["sim_s"] <= first["applied_sim_s"]), states[0])
            position = np.asarray(before["eef_position_m"])
            pre_speed = float(np.linalg.norm(before["measured_eef_velocity_m_s"]))
            direction = {"retreat": [-.06,0,.04], "lift":[0,0,1], "left":[0,-1,0], "right":[0,1,0]}.get(first["action"])
            if direction is not None:
                direction = np.asarray(direction); direction = direction/np.linalg.norm(direction)
            row["pre_command_speed_m_s"] = pre_speed
            row["pre_command_projected_velocity_m_s"] = float(np.dot(before["measured_eef_velocity_m_s"],direction)) if direction is not None else None
            streak = 0
            for state in states:
                if not first["applied_sim_s"] < state["sim_s"] <= min(first["applied_sim_s"]+1.5,event["end_sim_s"]):
                    continue
                if first["action"] == "brake":
                    reached = pre_speed > .02 and np.linalg.norm(state["measured_eef_velocity_m_s"]) <= min(.015,pre_speed/2)
                else:
                    reached = float(np.dot(np.asarray(state["eef_position_m"])-position,direction)) >= .005
                streak = streak+1 if reached else 0
                if streak >= 3:
                    row["command_to_motion_s"] = state["sim_s"]-first["applied_sim_s"]
                    row["onset_to_motion_s"] = state["sim_s"]-event["onset_sim_s"]
                    row["motion_sim_s"] = state["sim_s"]
                    row["command_to_motion_wall_s"] = state.get("measured_wall_s",state["wall_s"])-first["applied_wall_s"]
                    break
            if first["action"] == "brake" and pre_speed <= .02:
                row["motion_status"] = "already_slow_before_command"
            else:
                row["motion_status"] = "measured" if row["command_to_motion_s"] is not None else "criterion_not_reached_within_window"
        else:
            row["motion_status"] = "no_matched_new_command"
        rows.append(row)
    return rows
