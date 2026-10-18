"""Does the camera's "fast" mode (shrink the crop to 400 px, recognition only) hold up on full-size crops?

usage: python bench_fast_mode.py
Same 50 easy photos and 42 large 'y' photos as bench_ocr_fullsize.py, so the numbers can be compared with its
results: current reader 84% / 43%, recognition-only 80% / 26%.
"""
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import openpyxl
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from plate_ocr import PlateReader, clean_plate  # noqa: E402

ds = ROOT / "dataset-indian"
norm = lambda n: re.sub(r"[ ()_]", "", Path(n).stem).lower()
truth = {norm(r[0]): str(r[1]).upper().replace(" ", "") for r in
         openpyxl.load_workbook(ds / "number_plate.xlsx", read_only=True).active.iter_rows(min_row=2, values_only=True) if r[0] and r[1]}
allrows = list(csv.DictReader(open(ds / "eval_results.csv")))
easy = [r["image"] for r in allrows if r["detected"] == "1" and re.match(r"[A-Z]{2}\d", r["image"])]
hard = [r["image"] for r in allrows if r["image"].lower().startswith("y")]
easy = easy[::max(1, len(easy) // 50)][:50]
hard = hard[::max(1, len(hard) // 50)][:50]
det = YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt"))


def crops_of(names, imgsz):
    out = []
    for n in names:
        img = cv2.imread(str(ds / "images" / n))
        h, w = img.shape[:2]
        b = det.predict(img, imgsz=min(imgsz, max(640, -(-max(h, w) // 32) * 32)), conf=0.25, device="cpu", verbose=False)[0].boxes
        if len(b):
            x1, y1, x2, y2 = map(int, b[int(b.conf.argmax())].xyxy[0])
            out.append((img[y1:y2, x1:x2], truth[norm(n)]))
    return out


groups = {"easy photos": crops_of(easy, 1280), "large 'y' photos": crops_of(hard, 1920)}
r = PlateReader()
r.read(groups["easy photos"][0][0], mode="fast")
print(f"{'group':20} {'crops':>5} {'exact':>11} {'ms/plate':>9}   (crop widths: median)")
for g, c in groups.items():
    ok, t0 = 0, time.time()
    for crop, t in c:
        ok += clean_plate(r.read(crop, mode="fast")[0])[0] == t
    widths = sorted(x[0].shape[1] for x in c)
    print(f"{g:20} {len(c):5} {ok:3}/{len(c)} {100 * ok / len(c):3.0f}% {(time.time() - t0) / len(c) * 1000:8.0f}   ({widths[len(widths) // 2]} px)", flush=True)
