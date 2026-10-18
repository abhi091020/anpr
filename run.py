"""PPE + ANPR on a video file or RTSP stream.

usage: python run.py --source input.mp4 --out output.mp4 [--every 2] [--show]
"""
import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from plate_ocr import IN_PLATE, PlateReader, clean_plate

ROOT = Path(__file__).parent
PPE_MODEL = ROOT / "models/ppe/best.pt"
PLATE_MODEL = ROOT / "models/plate/license-plate-finetune-v1n.pt"

REQUIRED = ["helmet", "Vest", "goggles", "safety_shoe"]  # items checked per frame
COLORS = {"ppe": (0, 200, 0), "plate": (255, 160, 0)}

def read_plate(reader, crop):
    return reader.read(crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path, image path or rtsp:// url")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--every", type=int, default=2, help="run models on every Nth frame (CPU speed)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ppe-conf", type=float, default=0.4)
    ap.add_argument("--plate-conf", type=float, default=0.4)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    ppe = YOLO(str(PPE_MODEL))
    plate = YOLO(str(PLATE_MODEL))
    reader = PlateReader()

    is_image = Path(a.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    cap = None if is_image else cv2.VideoCapture(a.source)
    if cap is not None and not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25) if cap else 1
    writer = None
    votes = defaultdict(Counter)  # approximate-position track -> text votes
    log_rows, last_ppe, last_plates = [], [], []
    n, t0 = 0, time.time()

    while True:
        if is_image:
            frame = cv2.imread(a.source)
            ok = frame is not None
            if n:
                break
        else:
            ok, frame = cap.read()
        if not ok:
            break

        if n % a.every == 0:
            last_ppe = []
            r = ppe.predict(frame, imgsz=a.imgsz, conf=a.ppe_conf, device="cpu", verbose=False)[0]
            for b in r.boxes:
                last_ppe.append((ppe.names[int(b.cls)], float(b.conf), *map(int, b.xyxy[0])))
            last_plates = []
            r = plate.predict(frame, imgsz=a.imgsz, conf=a.plate_conf, device="cpu", verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                text, conf = read_plate(reader, frame[max(y1, 0):y2, max(x1, 0):x2])
                text, valid = clean_plate(text)
                key = (x1 // 80, y1 // 80)  # coarse position key so repeated reads of one plate vote together
                if text:
                    votes[key][text] += 1 + (2 if valid else 0)
                    text = votes[key].most_common(1)[0][0]
                    valid = bool(IN_PLATE.match(text))
                last_plates.append((text, valid, float(b.conf), x1, y1, x2, y2))
                if text:
                    log_rows.append((n, round(n / fps, 2), text, valid))

        seen = {c for c, *_ in last_ppe}
        for cls, conf, x1, y1, x2, y2 in last_ppe:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["ppe"], 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["ppe"], 1)
        for text, valid, conf, x1, y1, x2, y2 in last_plates:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["plate"], 2)
            label = text or "plate"
            cv2.putText(frame, label + ("" if valid else " ?"), (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["plate"], 2)
        missing = [r for r in REQUIRED if r not in seen] if any(c == "Person" or c in REQUIRED for c in seen) else []
        if missing:
            cv2.putText(frame, "NOT SEEN: " + ", ".join(missing), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if is_image:
            cv2.imwrite(a.out, frame)
        else:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if a.show:
            cv2.imshow("ppe-anpr", frame)
            if cv2.waitKey(1) == 27:
                break
        n += 1
        if n % 50 == 0:
            print(f"{n} frames, {n / (time.time() - t0):.1f} fps")

    if cap:
        cap.release()
    if writer:
        writer.release()
    csv_path = Path(a.out).with_suffix(".plates.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "seconds", "plate", "valid_indian_format"])
        w.writerows(log_rows)
    print(f"done: {n} frames in {time.time() - t0:.0f}s -> {a.out}, {csv_path}")


if __name__ == "__main__":
    main()
