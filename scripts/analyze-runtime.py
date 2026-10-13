"""Verify and summarize fixed-weight OmniJev / visual-brain paired evidence."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def percentile(values, p):
    if not values:
        return None
    a = sorted(values); i = (len(a)-1)*p; lo = int(i)
    return a[lo]+(a[min(lo+1, len(a)-1)]-a[lo])*(i-lo)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path)
    args = parser.parse_args(); root = args.root
    summary = json.loads((root/"summary.json").read_text())
    indexed = {(r["scenario"], r["seed"], r["mode"]): r for r in summary}
    assert len(indexed) == len(summary), "Duplicate identity"
    modes = ("omni", "vlm_omni")
    pairs = sorted({(r["scenario"], r["seed"]) for r in summary})
    detail, events, negatives, groups = [], [], [], {}
    for kind, seed in pairs:
        pair = [indexed[(kind, seed, mode)] for mode in modes]
        for key in ("initial_state_sha256", "scenario_parameters", "nominal_motor_plan", "software_limits"):
            assert pair[0][key] == pair[1][key], (kind, seed, key)
        row = {"scenario": kind, "seed": seed}
        for r in pair:
            mode = r["mode"]; directory = root/f"{kind}_seed{seed}_{mode}"
            assert json.loads((directory/"result.json").read_text()) == r
            row[mode] = {k: r[k] for k in ("final_goal_reached", "final_goal_distance_m", "peak_human_contact_N", "peak_object_contact_N", "human_contact_impulse_Ns", "object_contact_impulse_Ns", "watchdog_ticks", "control_steps")}
            row[mode]["matched_events"] = sum(e["first_action"] is not None for e in r["reactions"])
            row[mode]["events"] = len(r["reactions"])
            events += [{"mode": mode, "scenario": kind, "seed": seed, **e} for e in r["reactions"]]
        detail.append(row)
    for mode in modes:
        episodes = [r for r in summary if r["mode"] == mode]; apps, brain = [], []
        for r in episodes:
            d = root/f"{r['scenario']}_seed{r['seed']}_{mode}"
            a, b = rows(d/"applications.jsonl"), rows(d/"brain_applications.jsonl")
            for records, folder in ((a, "observations"), (b, "brain_observations")):
                for record in records:
                    if record.get("image_sha256"):
                        image = d/folder/f"{record['sequence']:06d}.jpg"
                        assert hashlib.sha256(image.read_bytes()).hexdigest() == record["image_sha256"], image
            apps += a; brain += b
            if r["scenario"] == "quiet":
                initial = next(x for x in json.loads((d/"initial_bank.json").read_text())["reflexes"] if x["id"] == "retreat")
                for record in a:
                    definition = next((x for x in record.get("bank_snapshot", {}).get("reflexes", []) if x["id"] == "retreat"), None)
                    if definition and definition["condition"] == initial["condition"] and record["observed_sim_s"] >= 2:
                        negatives.append({"mode": mode, "seed": r["seed"], "sequence": record["sequence"], "p": record["condition_probabilities"]["retreat"], "tau": definition["threshold"]})
        accepted = [a for a in apps if a["accepted"]]
        groups[mode] = {"episodes": len(episodes), "final_goal_count": sum(r["final_goal_reached"] for r in episodes),
                        "matched_events": sum(e["first_action"] is not None for e in events if e["mode"] == mode),
                        "events": sum(len(r["reactions"]) for r in episodes), "api_results": len(apps), "accepted_results": len(accepted),
                        "api_errors": sum("error" in a for a in apps), "error_types": dict(Counter(a["error"] for a in apps if "error" in a)),
                        "inference_p50_s": percentile([a["inference_s"] for a in apps], .5), "inference_p95_s": percentile([a["inference_s"] for a in apps], .95),
                        "capture_to_motor_p50_s": percentile([a["observation_age_s"] for a in accepted], .5), "capture_to_motor_p95_s": percentile([a["observation_age_s"] for a in accepted], .95),
                        "mean_watchdog_fraction": statistics.mean(r["watchdog_ticks"]/r["control_steps"] for r in episodes),
                        "brain_results": len(brain), "brain_accepted": sum(a["accepted"] for a in brain), "brain_errors": sum("error" in a for a in brain),
                        "brain_inference_p50_s": percentile([a["inference_s"] for a in brain], .5),
                        "macro_plan_changes": sum(r["macro_plan_changes"] for r in episodes), "bank_versions": sum(r["reflex_bank_versions"] for r in episodes),
                        "memory_entries": sum(r["memory_entries"] for r in episodes), "requests_with_retrieval": sum(bool(a.get("retrieved_memories")) for a in apps),
                        "served_models": sorted({a["served_model"] for a in apps if a.get("served_model")}), "brain_served_models": sorted({a["served_model"] for a in brain if a.get("served_model")})}
    bins = []
    for i in range(10):
        sample = [s for s in negatives if min(9, int(s["p"]*10)) == i]
        if sample: bins.append({"range": [i/10, (i+1)/10], "count": len(sample), "mean_p": statistics.mean(s["p"] for s in sample), "observed_frequency": 0})
    calibration = {"condition": "human-close retreat", "label": 0, "cohort": "quiet after 2s settling, initial condition unchanged; human outside camera",
                   "samples": len(negatives), "brier": statistics.mean(s["p"]**2 for s in negatives) if negatives else None,
                   "ece_10bins": sum(b["count"]*b["mean_p"] for b in bins)/len(negatives) if negatives else None,
                   "threshold_false_positive_fraction": statistics.mean(s["p"] >= s["tau"] for s in negatives) if negatives else None,
                   "bins": bins, "detail": negatives, "scope": "negative-cohort false-alarm check; not balanced calibration; frames correlated"}
    result = {"episodes": len(summary), "pairs": len(pairs), "groups": groups, "pair_detail": detail, "events_detail": events,
              "rtf_min": min(r["real_time_factor"] for r in summary), "rtf_max": max(r["real_time_factor"] for r in summary), "negative_calibration": calibration}
    (root/"metrics.json").write_text(json.dumps(result, indent=2))
    flat = [{"scenario": r["scenario"], "seed": r["seed"], **{mode+"_"+key: value for mode in modes for key, value in r[mode].items()}} for r in detail]
    with (root/"pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({"groups": groups, "episodes": len(summary), "pairs": len(pairs), "negative_samples": len(negatives)}, indent=2))


if __name__ == "__main__":
    main()
