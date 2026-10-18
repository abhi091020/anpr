"""Compare the readers through the app's real code path (two-size detection, padding, then reading).

usage: python bench_modes.py
50 easy photos + 50 large 'y' photos. For each plate it records what the full reader, the fast reader and the
"best of both" rule would give, so all three are scored on exactly the same detections.
"""
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import openpyxl

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from app.engine import engine  # noqa: E402
from plate_ocr import clean_plate, pick_better  # noqa: E402

ds = ROOT / "dataset-indian"
norm = lambda n: re.sub(r"[ ()_]", "", Path(n).stem).lower()
truth = {norm(r[0]): str(r[1]).upper().replace(" ", "") for r in
         openpyxl.load_workbook(ds / "number_plate.xlsx", read_only=True).active.iter_rows(min_row=2, values_only=True) if r[0] and r[1]}
rows = list(csv.DictReader(open(ds / "eval_results.csv")))
easy = [r["image"] for r in rows if re.match(r"[A-Z]{2}\d", r["image"])]
hard = [r["image"] for r in rows if r["image"].lower().startswith("y")]
groups = {"easy photos": easy[::max(1, len(easy) // 50)][:50], "large 'y' photos": hard[::max(1, hard.__len__() // 50)][:50]}
engine._photo_models()
stats = {}
for g, names in groups.items():
    s = {"found": 0, "full": 0, "fast": 0, "best": 0, "early": 0, "n": len(names), "t_full": 0.0, "t_fast": 0.0, "t_early": 0.0}
    for n in names:
        img = cv2.imread(str(ds / "images" / n))
        h, w = img.shape[:2]
        big = int(min(1920, max(640, -(-max(h, w) // 32) * 32)))
        boxes = engine._find_plates(img, [640] if big <= 640 else [640, big])[:1]
        if not boxes:
            continue
        s["found"] += 1
        t = truth[norm(n)]
        _, ocr = engine._photo_models()
        x1, y1, x2, y2, _c = boxes[0]
        px, py = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        crop = img[max(int(y1 - py), 0):min(int(y2 + py), h), max(int(x1 - px), 0):min(int(x2 + px), w)]
        t0 = time.time(); full = ocr.read(crop, mode="full"); tfull_ = time.time() - t0; s["t_full"] += tfull_
        t0 = time.time(); fast = ocr.read(crop, mode="fast"); tf_ = time.time() - t0; s["t_fast"] += tf_
        s["full"] += clean_plate(full[0])[0] == t
        s["fast"] += clean_plate(fast[0])[0] == t
        s["best"] += clean_plate(pick_better(full, fast)[0])[0] == t
        stop = clean_plate(fast[0])[1] and fast[1] >= 0.9  # fast reader already gave a confident, valid plate: skip the slow one
        s["early"] += clean_plate((fast if stop else pick_better(full, fast))[0])[0] == t
        s["t_early"] += tf_ + (0 if stop else tfull_)
    stats[g] = s
    f = max(s["found"], 1)
    print(f"{g}: plate found in {s['found']}/{s['n']}")
    for k in ("full", "fast", "best", "early"):
        ms = (s["t_full"] if k == "full" else s["t_fast"] if k == "fast" else s["t_early"] if k == "early" else s["t_full"] + s["t_fast"]) / f * 1000
        print(f"   {k:5}: {s[k]:3}/{s['found']} exact ({100 * s[k] / f:3.0f}% of the plates found)   {ms:6.0f} ms/plate (reading only)", flush=True)
