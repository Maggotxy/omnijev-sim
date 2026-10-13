"""Derive a complete paired results report from saved evidence; no model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def quantile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1)*p
    lower = int(position)
    upper = min(len(ordered)-1, lower+1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--title", default="连续反射实验：2026-10-13")
    args = parser.parse_args()
    summaries = json.loads((args.root/"summary.json").read_text())
    indexed = {(r["scenario"], r["seed"], r["mode"]): r for r in summaries}
    models = sorted({r["model"] for r in summaries if r["mode"] == "reflex"})
    if len(indexed) != len(summaries):
        raise ValueError("Duplicate episode identity")
    pairs = sorted({(r["scenario"], r["seed"]) for r in summaries})
    rows, event_rows, prefixes, latencies, applied_ages = [], [], [], [], []
    requests, accepted, errors = 0, 0, 0
    for kind, seed in pairs:
        base, reflex = [indexed[(kind, seed, mode)] for mode in ("baseline", "reflex")]
        for key in ("initial_state_sha256", "scenario_parameters", "nominal_motor_plan", "software_limits"):
            if base[key] != reflex[key]:
                raise ValueError(f"Pair mismatch: {kind}/{seed}/{key}")
        for result in (base, reflex):
            saved = json.loads((args.root/f"{kind}_seed{seed}_{result['mode']}"/"result.json").read_text())
            if saved != result:
                raise ValueError("Summary does not match episode result")
        directory = args.root/f"{kind}_seed{seed}_reflex"
        states = [json.loads(line) for line in (directory/"states.jsonl").read_text().splitlines()]
        applications = [json.loads(line) for line in (directory/"applications.jsonl").read_text().splitlines()]
        prefix = states if kind == "quiet" else [s for s in states if s["sim_s"] < reflex["scenario_parameters"]["onset"]]
        prefix_interventions = sum(s["action"] not in ("continue", "watchdog_hold") for s in prefix)
        if prefix_interventions != reflex["clean_prefix_model_intervention_ticks"]:
            raise ValueError("Prefix intervention count mismatch")
        prefixes.append({"scenario": kind, "seed": seed, "ticks": len(prefix),
                         "model_intervention_ticks": prefix_interventions,
                         "fraction": prefix_interventions/len(prefix) if prefix else None})
        for record in applications:
            if record.get("image_sha256"):
                image = directory/"observations"/record.get("image_file", f"{record['sequence']:06d}.jpg")
                if sha(image) != record["image_sha256"]:
                    raise ValueError("Model input JPEG hash mismatch")
            if record["accepted"]:
                applied_ages.append(record["observation_age_s"])
        requests += reflex["api_results"]
        accepted += reflex["accepted_results"]
        errors += reflex["api_errors"]
        latencies.extend(reflex["latency_wall_s"])
        reactions = reflex["reactions"]
        for event in reactions:
            event_rows.append({"scenario": kind, "seed": seed, **event})
        responses = [e for e in reactions if e["first_action"]]
        rows.append({"scenario": kind, "seed": seed,
                     "baseline_final_goal": base["final_goal_reached"], "reflex_final_goal": reflex["final_goal_reached"],
                     "baseline_goal_error_cm": base["final_goal_distance_m"]*100,
                     "reflex_goal_error_cm": reflex["final_goal_distance_m"]*100,
                     "baseline_human_peak_N": base["peak_human_contact_N"], "reflex_human_peak_N": reflex["peak_human_contact_N"],
                     "baseline_object_peak_N": base["peak_object_contact_N"], "reflex_object_peak_N": reflex["peak_object_contact_N"],
                     "baseline_human_impulse_Ns": base["human_contact_impulse_Ns"], "reflex_human_impulse_Ns": reflex["human_contact_impulse_Ns"],
                     "matched_response_events": len(responses), "events": len(reactions),
                     "prefix_intervention_fraction": prefixes[-1]["fraction"],
                     "watchdog_fraction": reflex["watchdog_ticks"]/reflex["control_steps"],
                     "accepted_results": reflex["accepted_results"], "api_errors": reflex["api_errors"]})
    result = {"episodes": len(summaries), "pairs": len(pairs), "events": len(event_rows),
              "matched_response_events": sum(e["first_action"] is not None for e in event_rows),
              "baseline_final_goal_count": sum(r["mode"] == "baseline" and r["final_goal_reached"] for r in summaries),
              "reflex_final_goal_count": sum(r["mode"] == "reflex" and r["final_goal_reached"] for r in summaries),
              "api_results": requests, "accepted_results": accepted, "api_errors": errors,
              "inference_p50_s": quantile(latencies, .5), "inference_p95_s": quantile(latencies, .95),
              "accepted_capture_to_motor_p50_s": quantile(applied_ages, .5),
              "accepted_capture_to_motor_p95_s": quantile(applied_ages, .95),
              "reaction_capture_to_motor_s": [e["observation_to_motor_wall_s"] for e in event_rows if e["first_action"]],
              "reaction_from_onset_s": [e["response_from_onset_sim_s"] for e in event_rows if e["first_action"]],
              "rtf_min": min(r["real_time_factor"] for r in summaries),
              "rtf_max": max(r["real_time_factor"] for r in summaries),
              "pairs_detail": rows, "events_detail": event_rows, "prefixes_detail": prefixes,
              "mean_reflex_watchdog_fraction": statistics.mean(r["watchdog_ticks"]/r["control_steps"] for r in summaries if r["mode"] == "reflex"),
              "served_models": sorted({json.loads(line).get("served_model") for kind, seed in pairs
                                         for line in (args.root/f"{kind}_seed{seed}_reflex"/"api.jsonl").read_text().splitlines()
                                         if json.loads(line).get("served_model")})}
    (args.root/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(args.root/"pairs.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    root_link = "../results/"+args.root.name
    lines = ["# "+args.title, "",
             f"完成 **{len(summaries)} 轮、{len(pairs)} 组成对试验**。每轮 20 秒连续物理，seeds 73、97 未用于前期提示调试。六类情况为无干扰、人手、滑动物体、目标移动、遮挡及随机顺序组合。",
             "", f"符合预设类别且真正改变电机命令的事件后响应为 **{result['matched_response_events']}/{result['events']}**。原计划对照组最终到达目标 **{result['baseline_final_goal_count']}/{len(pairs)}**；模型反射组 **{result['reflex_final_goal_count']}/{len(pairs)}**。到达目标与避让效果必须分别看待。",
             "", "## 时序与接口", "",
             f"- 世界实时因子 {result['rtf_min']:.6f}–{result['rtf_max']:.6f}，物理未等待模型。",
             f"- 决策调用完成 {requests} 次，接受 {accepted} 次，错误 {errors} 次；结束后返回的答复及过期答复不应用。",
             f"- 推理耗时 p50 **{result['inference_p50_s']*1000:.0f} ms**，p95 **{result['inference_p95_s']*1000:.0f} ms**；包括超时请求。",
             f"- 已接受答复的采集到命令应用延迟 p50 **{result['accepted_capture_to_motor_p50_s']*1000:.0f} ms**，p95 **{result['accepted_capture_to_motor_p95_s']*1000:.0f} ms**。",
             f"- 模型组平均 **{result['mean_reflex_watchdog_fraction']:.1%}** 的控制周期处于软件过期保护。它不属于模型反射。",
             f"- 模型 `{', '.join(models)}`；执行方模型标识：`{', '.join(result['served_models']) or '未提供'}`。本地标识由本地执行器记录，云端标识来自服务端头。",
             "", "## 全部配对结果", "",
             "人手/物体接触列为整轮峰值，单位 N；左右分别为原计划对照/模型反射。干扰前干预是无干扰前缀中非继续、非 watchdog 命令的周期比例，quiet 使用整轮。",
             "", "| 情况 / seed | 最终到目标 对照/模型 | 目标误差 cm 对照/模型 | 人手峰值 对照/模型 | 物体峰值 对照/模型 | 事件后响应 | 干扰前干预 | watchdog | 视频 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        kind, seed = row["scenario"], row["seed"]
        yes = lambda x: "是" if x else "否"
        lines.append(f"| {kind} / {seed} | {yes(row['baseline_final_goal'])}/{yes(row['reflex_final_goal'])} | "
                     f"{row['baseline_goal_error_cm']:.1f}/{row['reflex_goal_error_cm']:.1f} | "
                     f"{row['baseline_human_peak_N']:.1f}/{row['reflex_human_peak_N']:.1f} | "
                     f"{row['baseline_object_peak_N']:.1f}/{row['reflex_object_peak_N']:.1f} | "
                     f"{row['matched_response_events']}/{row['events']} | {row['prefix_intervention_fraction']:.1%} | "
                     f"{row['watchdog_fraction']:.1%} | [成对录像]({root_link}/comparison_{kind}_seed{seed}.mp4) |")
    lines += ["", "## 每个事件的首个命令响应", "",
              "| 情况 / seed | 事件 | 开始 s | 首动作 | 开始到命令 s | 采集到命令 ms |",
              "|---|---|---|---|---|---|"]
    for event in event_rows:
        delay = f"{event['response_from_onset_sim_s']:.3f}" if event['first_action'] else "—"
        age = f"{event['observation_to_motor_wall_s']*1000:.0f}" if event['first_action'] else "—"
        lines.append(f"| {event['scenario']} / {event['seed']} | {event['kind']} | {event['onset_sim_s']:.3f} | "
                     f"{event['first_action'] or '未记录匹配响应'} | {delay} | {age} |")
    lines += ["", "## 解释与限制", "",
              "响应表使用事先定义的动作类别，只证明事件之后发生了匹配的命令转换。尤其在重叠干扰中，同一个命令可能匹配多个事件，不能据此判断模型识别了哪个事件。独立事件与组合事件分别列出。实际末端速度保存于 states.jsonl，可检查后续机械运动。",
              "", "干扰前误退让可能提前让机器人远离接触区域，因此接触力下降本身不能证明正确条件反射。错误动作、持续保持、任务失败和 watchdog 均保留。当前云端模型的可靠性和延迟仍限制反射效果。",
              "", "配对按同一 Scenario、初始状态哈希、原运动计划和软件限制核验；执行顺序固定为对照后模型，没有顺序平衡。试验是简化刚体场景的小样本，不能外推到全部日常情况、毫秒级人类反射或真人安全。",
              "", "## 可复核材料", "",
              f"- [原始汇总]({root_link}/summary.json)、[机器可读分析]({root_link}/metrics.json)、[配对 CSV]({root_link}/pairs.csv)。",
              f"- [运行源码哈希和参数]({root_link}/manifest.json)、[机器人网格哈希]({root_link}/mesh_manifest.json)、[录像参数]({root_link}/videos.json)。",
              "- 每轮目录包含完整 scene.xml、scenario.json、states.jsonl、applications.jsonl、api.jsonl，以及每次实际发送 JPEG；分析器核验模型记录的 JPEG 哈希。",
              "- 机械/异步测试通过，测试中固定响应明确标为测试替身，不作为模型效果证据。",
              "- 浏览器交互另有 [验收回执](../results/reflex_ui_20261013/qa_receipt.json)；它核验按钮与持久化，不证明模型普遍正确。",
              f"- 使用 `python scripts/summarize-reflex.py results/{args.root.name} --report {args.report.as_posix()} --title '{args.title}'` 可重新生成此报告。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
