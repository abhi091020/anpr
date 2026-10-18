"""Cut a plate-crop training set out of the labelled photos: ocr-dataset/{train,val,test}/*.png + labels.csv.

usage: python make_ocr_dataset.py

Split (stable, by md5 of the file name, same rule as eval_plates.py):
  test  = md5 % 10 >= 6   (40%, never trained on, used only to measure accuracy)
  val   = md5 % 10 == 5   (10%, picks the best training epoch)
  train = md5 % 10 <= 4   (50%)
The plate is found with the plate detector, so training sees the same kind of crop the live system produces.
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

import cv2
import openpyxl
from ultralytics import YOLO

ROOT = Path(__file__).parent
ds = ROOT / "dataset-indian"
out = ROOT / "ocr-dataset"
PAD = 0.04  # grow the detector box slightly so edge characters are not clipped


def norm_key(name):
    return re.sub(r"[ ()_]", "", Path(name).stem).lower()


def split_of(name):
    m = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    return "test" if m >= 6 else "val" if m == 5 else "train"


truth = {norm_key(r[0]): str(r[1]).upper().replace(" ", "") for r in
         openpyxl.load_workbook(ds / "number_plate.xlsx", read_only=True).active.iter_rows(min_row=2, values_only=True)
         if r[0] and r[1]}
det = YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt"))

for s in ("train", "val", "test"):
    (out / s).mkdir(parents=True, exist_ok=True)
rows, skipped = [], []
files = sorted(p for p in (ds / "images").iterdir() if norm_key(p.name) in truth)
for i, img_path in enumerate(files, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        skipped.append((img_path.name, "unreadable"))
        continue
    boxes = det.predict(img, imgsz=640, conf=0.25, device="cpu", verbose=False)[0].boxes
    if not len(boxes):
        skipped.append((img_path.name, "no plate found"))
        continue
    b = boxes[int(boxes.conf.argmax())]
    x1, y1, x2, y2 = map(float, b.xyxy[0])
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    h, w = img.shape[:2]
    crop = img[max(int(y1 - ph), 0):min(int(y2 + ph), h), max(int(x1 - pw), 0):min(int(x2 + pw), w)]
    split = split_of(img_path.name)
    name = f"{split}/{img_path.stem.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    cv2.imwrite(str(out / name), crop)
    rows.append({"file": name, "text": truth[norm_key(img_path.name)], "split": split, "source": img_path.name})
    if i % 200 == 0:
        print(f"{i}/{len(files)}", flush=True)

with open(out / "labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "text", "split", "source"])
    w.writeheader()
    w.writerows(rows)
counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
print("crops:", len(rows), counts, "| skipped:", len(skipped), skipped[:5])
