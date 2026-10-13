"""Pair recordings on wall time, validate scenario equality, label model reactions."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .run import FPS, font


def verify_pair(base, reflex):
    for key in ("scenario", "seed", "initial_state_sha256", "scenario_parameters", "nominal_motor_plan", "software_limits"):
        if base[key] != reflex[key]:
            raise ValueError(f"Pair mismatch: {key}")
    if (base["mode"], reflex["mode"]) not in (("baseline", "reflex"), ("omni", "vlm_omni")):
        raise ValueError("Unknown comparison pair")


def compose(root, kind, seed, modes=("baseline", "reflex")):
    directories = [root/f"{kind}_seed{seed}_{mode}" for mode in modes]
    results = [json.loads((p/"result.json").read_text()) for p in directories]
    verify_pair(*results)
    readers = [imageio.get_reader(str(p/"episode.mp4")) for p in directories]
    counts = [reader.count_frames() for reader in readers]
    path = root/f"comparison_{kind}_seed{seed}.mp4"
    samples, last = [], [None, None]
    count = max(counts)
    try:
        with imageio.get_writer(str(path), fps=FPS, quality=7) as writer:
            for index in range(count):
                im = Image.new("RGB", (1920, 800), (7, 12, 22))
                draw = ImageDraw.Draw(im)
                draw.text((20, 8), f"CONTINUOUS REFLEX | {kind} | seed {seed} | paired external disturbance", fill=(238, 244, 250), font=font(27))
                reaction = next((r for r in results[1]["reactions"] if r["first_action"]), None)
                summary = "No matched event response recorded in right panel"
                if reaction:
                    summary = (f"Model {reaction['first_action']} @ {reaction['first_model_reaction_sim_s']:.2f}s | "
                               f"capture-to-motor {reaction['observation_to_motor_wall_s']*1000:.0f} ms")
                draw.text((20, 46), f"Wall-time playback 1x | {summary}", fill=(137, 223, 179), font=font(21))
                for column, reader in enumerate(readers):
                    if index < counts[column]:
                        last[column] = reader.get_data(index)
                    im.paste(Image.fromarray(last[column]), (column*960, 80))
                writer.append_data(np.asarray(im))
                event_index = min(count-1, int((results[0]["scenario_parameters"]["onset"]+1)*FPS))
                if index in (0, event_index, count-1):
                    samples.append(im.copy())
                if index % 200 == 0:
                    print(f"Compose {kind} seed={seed} {index}/{count}", flush=True)
    finally:
        for reader in readers:
            reader.close()
    sheet = Image.new("RGB", (1920, 800*len(samples)))
    for i, sample in enumerate(samples):
        sheet.paste(sample, (0, i*800))
    sheet.save(root/f"comparison_{kind}_seed{seed}_contact.png")
    return {"scenario": kind, "seed": seed, "video": path.name, "frames": count, "fps": FPS,
            "width": 1920, "height": 800, "duration_s": count/FPS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summaries = json.loads((args.root/"summary.json").read_text())
    modes = ("omni", "vlm_omni") if {r["mode"] for r in summaries} == {"omni", "vlm_omni"} else ("baseline", "reflex")
    pairs = sorted({(r["scenario"], r["seed"]) for r in summaries})
    videos = [compose(args.root, kind, seed, modes) for kind, seed in pairs]
    (args.root/"videos.json").write_text(json.dumps(videos, indent=2))


if __name__ == "__main__":
    main()
