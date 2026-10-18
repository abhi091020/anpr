"""Does the recognition-only reader hold up on full-size plate crops (photos), not only webcam-size ones?

usage: python bench_ocr_fullsize.py
Compares the current reader with recognition-only (medium model) on original-resolution crops from two groups:
the easy state-code photos, and the large 'y' photos where the plate is a small strip.
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
hard = [r["image"] for r in allrows if r["image"].lower().startswith("y")]  # large photos; many were missed at 640
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


groups = {"easy photos": (crops_of(easy, 1280), len(easy)), "large 'y' photos": (crops_of(hard, 1920), len(hard))}
for g, (c, n) in groups.items():
    print(f"{g}: plate found in {len(c)}/{n} (that part is the detector, same for both readers)", flush=True)

from paddleocr import TextRecognition
cur = PlateReader()
try:
    rec = TextRecognition(model_name="PP-OCRv6_medium_rec", enable_mkldnn=False)
except TypeError:
    rec = TextRecognition(model_name="PP-OCRv6_medium_rec")


def rec_only(crop):
    h, w = crop.shape[:2]
    parts = [crop] if w / h > 2.6 else [crop[: h // 2], crop[h // 2:]]
    return "".join(re.sub(r"[^A-Za-z0-9]", "", rec.predict(input=p, batch_size=1)[0]["rec_text"]).upper() for p in parts)


readers = {"current": lambda c: cur.read(c)[0], "recognition only": rec_only}
cur.read(groups["easy photos"][0][0][0]); rec_only(groups["easy photos"][0][0][0])  # warm-up
print(f"\n{'group':20} {'reader':18} {'exact':>11} {'ms/plate':>9}")
for g, (c, _) in groups.items():
    for rn, fn in readers.items():
        ok, t0 = 0, time.time()
        for crop, t in c:
            ok += clean_plate(fn(crop))[0] == t
        print(f"{g:20} {rn:18} {ok:3}/{len(c)} {100 * ok / len(c):3.0f}% {(time.time() - t0) / len(c) * 1000:8.0f}", flush=True)
