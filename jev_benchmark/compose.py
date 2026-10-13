"""Render side-by-side evidence; unavailable models remain explicit static panels."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from .run_suite import MODES, TITLES, IMG, FPS, font


def verify_pairing(results):
    executed = [r for r in results if r.get("initial_state_sha256")]
    modes = [r.get("mode") for r in results]
    if len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("Expected distinct comparison modes")
    required = ("seed", "init_index", "task_name", "initial_state_sha256", "settled_state_sha256", "controller", "limits")
    for key in required:
        if any(key not in r for r in executed):
            raise ValueError(f"Missing pairing metadata: {key}")
        if len({json.dumps(r[key], sort_keys=True) for r in executed}) > 1:
            raise ValueError(f"Cannot compare different {key}")
    return {r["mode"] for r in executed} == set(MODES)


def unavailable(mode, result, initial):
    canvas = Image.new("RGB", (IMG, 736), (15, 20, 30))
    if initial is not None:
        canvas.paste(ImageEnhance.Brightness(Image.fromarray(initial)).enhance(0.25), (0, 64))
    d = ImageDraw.Draw(canvas)
    d.text((16, 14), TITLES[mode], fill=(200, 210, 225), font=font(25))
    d.rectangle((24, 210, 488, 370), fill=(21, 28, 42), outline=(110, 120, 140), width=2)
    d.text((45, 236), "NOT RUN", fill=(255, 197, 114), font=font(32))
    d.text((45, 290), result["status"], fill=(230, 233, 241), font=font(20))
    code = "HTTP 404: R2 model unavailable" if "HTTP 404" in result.get("error", "") else "See result.json for the API error"
    d.text((32, 606), code, fill=(230, 233, 241), font=font(19))
    d.text((32, 644), "No episode / no success-rate sample", fill=(160, 172, 190), font=font(18))
    return np.asarray(canvas)


def compose(root, seed):
    directories = {mode: root / f"{mode}_seed{seed}" for mode in MODES}
    results = {mode: json.loads((path / "result.json").read_text()) for mode, path in directories.items()}
    if any(r.get("mode") != mode or r.get("seed") != seed for mode, r in results.items()):
        raise ValueError("Result metadata does not match the requested mode/seed directory")
    complete = verify_pairing(list(results.values()))
    initial_path = next((path / "initial.png" for path in directories.values() if (path / "initial.png").exists()), None)
    initial = np.asarray(Image.open(initial_path).convert("RGB")) if initial_path else None
    readers, counts, last = {}, {}, {}
    for mode, path in directories.items():
        if (path / "episode.mp4").exists():
            reader = imageio.get_reader(str(path / "episode.mp4"))
            readers[mode], counts[mode] = reader, reader.count_frames()
        else:
            last[mode] = unavailable(mode, results[mode], initial)
    output = root / f"comparison_seed{seed}.mp4"
    count = max(counts.values(), default=FPS * 6)
    samples = []
    try:
        with imageio.get_writer(str(output), fps=FPS, quality=7) as writer:
            for index in range(count):
                canvas = Image.new("RGB", (IMG * 3, 816), (9, 14, 23))
                d = ImageDraw.Draw(canvas)
                d.text((22, 8), "LIBERO-10 | Two objects into basket | Shared perception + control", fill=(238, 243, 251), font=font(25))
                status = "Paired comparison" if complete else "PARTIAL EVIDENCE: R2 access required for the full comparison"
                d.text((22, 45), f"{status} | seed {seed} | simulation-time playback at 1x", fill=(170, 185, 207), font=font(18))
                for col, mode in enumerate(MODES):
                    if mode in readers and index < counts[mode]:
                        last[mode] = readers[mode].get_data(index)
                    canvas.paste(Image.fromarray(last[mode]), (col * IMG, 80))
                writer.append_data(np.asarray(canvas))
                if index in (0, count // 2, count - 1):
                    samples.append(canvas.copy())
                if index % 200 == 0:
                    print(f"Compose seed={seed} frame={index}/{count}", flush=True)
    finally:
        for reader in readers.values():
            reader.close()
    sheet = Image.new("RGB", (IMG * 3, 816 * len(samples)))
    for index, im in enumerate(samples):
        sheet.paste(im, (0, 816 * index))
    sheet.save(root / f"comparison_seed{seed}_contact.png")
    return {"seed": seed, "complete_comparison": complete, "video": output.name, "frames": count,
            "width": 1536, "height": 816, "fps": FPS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7])
    args = parser.parse_args()
    outputs = [compose(args.root, seed) for seed in args.seeds]
    (args.root / "videos.json").write_text(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
