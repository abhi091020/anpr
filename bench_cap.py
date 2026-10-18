"""Can large plate crops be shrunk before the full reader without losing accuracy? (speeds up huge photos)

usage: python bench_cap.py
Uses the 'y' photos (large phone photos). Baseline measured earlier: full reader, no shrinking, 43% exact.
"""
import csv
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
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
hard = [r["image"] for r in csv.DictReader(open(ds / "eval_results.csv")) if r["image"].lower().startswith("y")]
hard = hard[::max(1, len(hard) // 50)][:50]
det = YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt"))
crops = []
for n in hard:
    img = cv2.imread(str(ds / "images" / n))
    b = det.predict(img, imgsz=1920, conf=0.25, device="cpu", verbose=False)[0].boxes
    if len(b):
        x1, y1, x2, y2 = map(int, b[int(b.conf.argmax())].xyxy[0])
        crops.append((img[y1:y2, x1:x2], truth[norm(n)]))
widths = sorted(c.shape[1] for c, _ in crops)
print(f"{len(crops)} large-photo crops, widths {widths[0]}..{widths[-1]} px (median {widths[len(widths) // 2]})\n", flush=True)
r = PlateReader()
r.read(crops[0][0], mode="full")
print(f"{'crop shrunk to (px wide)':26} {'exact':>11} {'ms/plate':>9}")
for cap in (None, 960, 720, 540):
    ok, t0 = 0, time.time()
    for c, t in crops:
        if cap and c.shape[1] > cap:
            c = cv2.resize(c, None, fx=cap / c.shape[1], fy=cap / c.shape[1], interpolation=cv2.INTER_AREA)
        ok += clean_plate(r.read(c, mode="full")[0])[0] == t
    print(f"{'no shrinking' if cap is None else cap:26} {ok:3}/{len(crops)} {100 * ok / len(crops):3.0f}% {(time.time() - t0) / len(crops) * 1000:8.0f}", flush=True)
