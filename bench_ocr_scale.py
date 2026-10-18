"""Speed vs accuracy of the OCR enlargement setting, on plate crops shrunk to webcam size.

usage: python bench_ocr_scale.py
Takes plates from the easy state-code photos, shrinks each crop to ~250 px wide (what a laptop camera gives for a
plate held up to it), and reads it at several enlargement targets.
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
rows = [r for r in csv.DictReader(open(ds / "eval_results.csv")) if r["detected"] == "1" and re.match(r"[A-Z]{2}\d", r["image"])]
rows = rows[::max(1, len(rows) // 40)][:40]
det = YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt"))
crops = []
for r in rows:
    img = cv2.imread(str(ds / "images" / r["image"]))
    b = det.predict(img, imgsz=1280, conf=0.25, device="cpu", verbose=False)[0].boxes
    if not len(b):
        continue
    x1, y1, x2, y2 = map(int, b[int(b.conf.argmax())].xyxy[0])
    c = img[y1:y2, x1:x2]
    c = cv2.resize(c, (250, max(1, int(c.shape[0] * 250 / c.shape[1]))), interpolation=cv2.INTER_AREA)  # webcam-size
    crops.append((c, truth[norm(r["image"])]))
print(f"{len(crops)} plate crops, shrunk to 250 px wide\n")
reader = PlateReader()
reader.read(crops[0][0])  # warm-up, so the first setting is not charged for start-up
for tw, n in ((750, 12), (480, len(crops)), (400, len(crops)), (320, len(crops)), (250, len(crops))):
    ok, t0 = 0, time.time()
    sub = crops[:n]
    for c, t in sub:
        raw, _ = reader.read(c, target_width=tw)
        ok += clean_plate(raw)[0] == t
    print(f"enlarge to {tw:4} px wide: {ok}/{len(sub)} exact ({100 * ok / len(sub):.0f}%), {(time.time() - t0) / len(sub) * 1000:.0f} ms per plate"
          + ("   <- the old behaviour (3x)" if tw == 750 else ""), flush=True)
