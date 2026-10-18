"""Speed vs accuracy of OCR variants on plate crops shrunk to webcam size.

usage: python bench_ocr_speed.py
All variants read the same 40 crops (easy state-code photos, each shrunk to ~300 px wide, like a laptop/phone camera
frame) and are scored by exact match against the correct plate, so a faster option is not chosen by speed alone.
"""
import csv
import re
import sys
import time
import traceback
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
    crops.append((cv2.resize(c, (300, max(1, int(c.shape[0] * 300 / c.shape[1]))), interpolation=cv2.INTER_AREA), truth[norm(r["image"])]))
print(f"{len(crops)} plate crops, ~300 px wide\n", flush=True)


def rec_only(model_name):
    """Recognition model only: no separate text-finding step, because the plate is already cropped."""
    from paddleocr import TextRecognition
    try:
        m = TextRecognition(model_name=model_name, enable_mkldnn=False)
    except TypeError:
        m = TextRecognition(model_name=model_name)

    def read(crop):
        h, w = crop.shape[:2]
        parts = [crop] if w / h > 2.6 else [crop[: h // 2], crop[h // 2:]]  # a squarer plate is two lines: read each
        text = ""
        for part in parts:
            out = m.predict(input=part, batch_size=1)
            text += re.sub(r"[^A-Za-z0-9]", "", out[0]["rec_text"]).upper()
        return text
    return read


def full_pipeline(det_name=None, rec_name=None):
    from paddleocr import PaddleOCR
    kw = dict(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)
    if det_name:
        kw.update(text_detection_model_name=det_name, text_recognition_model_name=rec_name)
    o = PaddleOCR(**kw)

    def read(crop):
        if crop.shape[1] < 480:
            s = min(3.0, 480 / crop.shape[1]); crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        out = o.predict(crop)[0]
        items = sorted(zip(out["rec_polys"], out["rec_texts"]), key=lambda t: (round(t[0][0][1] / 30), t[0][0][0]))
        return "".join(re.sub(r"[^A-Za-z0-9]", "", t).upper() for _, t in items)
    return read


variants = [
    ("current (medium det + rec)", lambda: (lambda r: (lambda crop: clean_plate(r.read(crop)[0])[0]))(PlateReader())),
    ("rec-only, medium model", lambda: (lambda f: (lambda crop: clean_plate(f(crop))[0]))(rec_only("PP-OCRv6_medium_rec"))),
    ("mobile det + rec (v5)", lambda: (lambda f: (lambda crop: clean_plate(f(crop))[0]))(full_pipeline("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"))),
    ("rec-only, mobile model (v5)", lambda: (lambda f: (lambda crop: clean_plate(f(crop))[0]))(rec_only("PP-OCRv5_mobile_rec"))),
]
print(f"{'variant':32} {'exact':>9} {'ms/plate':>9}")
for name, build in variants:
    try:
        read = build()
        read(crops[0][0])  # warm-up
        ok, t0 = 0, time.time()
        for c, t in crops:
            ok += read(c) == t
        ms = (time.time() - t0) / len(crops) * 1000
        print(f"{name:32} {ok:3}/{len(crops)} {100 * ok / len(crops):3.0f}% {ms:8.0f}", flush=True)
    except Exception as e:
        print(f"{name:32} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
