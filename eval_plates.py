"""Measure plate detection + OCR accuracy on the labelled photos, with a fixed tune/test split.

usage: python eval_plates.py [dataset-indian] [--limit N]

Split: a photo goes to "test" if md5(filename) % 10 >= 6, else "tune" (about 60/40, stable across runs).
Photos listed in seen_96_before_split.csv were used to tune the cleanup rules, so they are marked
seen=1 and left out of the "clean test" figure.

Writes <dataset>/eval_results.csv (with the raw OCR text so cleanup rules can be re-scored offline).
"""
import csv
import hashlib
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import openpyxl
from ultralytics import YOLO

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from plate_ocr import PlateReader, clean_plate  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
if limit is not None:
    args = [a for a in args if a != str(limit)]
ds = ROOT / (args[0] if args else "dataset-indian")

def norm_key(name):
    """The sheet and the image files disagree on spaces, brackets, case and extension ('C1.png' vs 'C (1).png',
    'Y859.png' vs 'y859.jpg'), so match on the name with those ignored (checked: no collisions)."""
    return re.sub(r"[ ()_]", "", Path(name).stem).lower()


truth = {norm_key(r[0]): str(r[1]).upper().replace(" ", "") for r in
         openpyxl.load_workbook(ds / "number_plate.xlsx", read_only=True).active.iter_rows(min_row=2, values_only=True)
         if r[0] and r[1]}
seen_file = ds / "seen_96_before_split.csv"
seen = {r["image"] for r in csv.DictReader(open(seen_file))} if seen_file.exists() else set()


def split_of(name):
    return "test" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 >= 6 else "tune"


def char_acc(a, b):
    """1 - normalised edit distance."""
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return 1 - d[-1] / max(len(a), len(b), 1)


def error_type(pred, t, detected):
    if pred == t:
        return "ok"
    if not detected:
        return "plate_not_found"
    if not pred:
        return "nothing_read"
    if len(pred) == len(t):
        return "wrong_chars"
    return "missing_or_extra_chars"


plate = YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt"))
reader = PlateReader()

files = sorted(p for p in (ds / "images").iterdir() if norm_key(p.name) in truth)
if limit:
    files = files[:limit]
fields = ["image", "split", "seen", "truth", "raw", "pred", "detected", "exact", "char_acc", "valid_format",
          "error", "det_s", "ocr_s"]
out = open(ds / "eval_results.csv", "w", newline="")
w = csv.DictWriter(out, fieldnames=fields)
w.writeheader()
rows = []
t_start = time.time()
for i, img in enumerate(files, 1):
    frame = cv2.imread(str(img))
    if frame is None:
        continue
    t0 = time.time()
    res = plate.predict(frame, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
    det_s = time.time() - t0
    raw, pred, valid, ocr_s = "", "", False, 0.0
    if len(res.boxes):
        b = res.boxes[int(res.boxes.conf.argmax())]
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        t1 = time.time()
        raw, _ = reader.read(frame[max(y1, 0):y2, max(x1, 0):x2])
        ocr_s = time.time() - t1
        pred, valid = clean_plate(raw)
    t = truth[norm_key(img.name)]
    detected = len(res.boxes) > 0
    row = {"image": img.name, "split": split_of(img.name), "seen": int(img.name in seen), "truth": t, "raw": raw,
           "pred": pred, "detected": int(detected), "exact": int(pred == t), "char_acc": round(char_acc(pred, t), 3),
           "valid_format": int(valid), "error": error_type(pred, t, detected), "det_s": round(det_s, 3),
           "ocr_s": round(ocr_s, 3)}
    rows.append(row)
    w.writerow(row)
    if i % 50 == 0:
        out.flush()
        print(f"{i}/{len(files)}  {time.time() - t_start:.0f}s", flush=True)
out.close()


def report(name, rs):
    n = len(rs)
    if not n:
        return
    print(f"\n== {name}: {n} photos")
    print(f"plate found       {sum(r['detected'] for r in rs)}/{n}  ({100 * sum(r['detected'] for r in rs) / n:.1f}%)")
    print(f"exact text        {sum(r['exact'] for r in rs)}/{n}  ({100 * sum(r['exact'] for r in rs) / n:.1f}%)")
    print(f"valid IN format   {sum(r['valid_format'] for r in rs)}/{n}")
    print(f"mean char acc     {sum(r['char_acc'] for r in rs) / n:.3f}")
    print(f"seconds/photo     detect {sum(r['det_s'] for r in rs) / n:.2f}  ocr {sum(r['ocr_s'] for r in rs) / n:.2f}")
    print("errors:", dict(Counter(r["error"] for r in rs if r["error"] != "ok")))


report("ALL", rows)
report("TUNE split", [r for r in rows if r["split"] == "tune"])
report("TEST split", [r for r in rows if r["split"] == "test"])
report("CLEAN TEST (test split, not used for tuning)", [r for r in rows if r["split"] == "test" and not r["seen"]])
