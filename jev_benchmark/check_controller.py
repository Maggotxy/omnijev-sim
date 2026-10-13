"""Mechanical acceptance test with a scripted plan; never model evidence."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .run_suite import Episode, TITLES


class Scripted:
    model = "scripted-controller-test"

    def __init__(self):
        self.records = []

    def plan(self, frame, task, candidates, completed):
        for obj in ("tomato sauce", "alphabet soup"):
            if obj not in completed:
                return {"object": obj, "target": "basket"}
        return {"object": "finish"}

    def gate(self, frame, context):
        return {"action": "proceed", "source": "scripted-controller-test"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    settings = SimpleNamespace(task_idx=0, init_index=0, max_gate_waits=1, max_plans=3, max_steps=2400)
    policy = Scripted()
    TITLES["omni"] = "CONTROLLER CHECK (scripted)"
    ep = Episode(settings, "omni", 1, args.out, {"omni": policy, "gemini": policy})
    result = ep.run()
    result["mode"] = "controller_check"
    result["model_evidence"] = False
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
