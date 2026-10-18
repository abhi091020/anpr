"""A plate is swapped for a different one at the SAME spot in the camera view: the app must notice and record the new one.

usage: python test_swap.py     (needs the app running on 8010)
Builds 1280x720 frames with plate A at a fixed place, then plate B at exactly the same place, and sends them like the
camera page does (one frame about every 0.3 s).
"""
import json
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent
BASE = "http://127.0.0.1:8010"
det = YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt"))


def crop_of(name):
    img = cv2.imread(str(ROOT / "dataset-indian/images" / name))
    b = det.predict(img, imgsz=640, conf=0.25, device="cpu", verbose=False)[0].boxes
    x1, y1, x2, y2 = map(int, b[int(b.conf.argmax())].xyxy[0])
    return img[y1:y2, x1:x2]


def frame_with(crop):
    f = np.full((720, 1280, 3), 110, np.uint8)
    f[300:300 + 120, 420:420 + 440] = cv2.resize(crop, (440, 120))  # the same spot every time
    return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def post(path, data=None):
    if data is None:
        req = urllib.request.Request(BASE + path, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    else:
        b = uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


A, B = frame_with(crop_of("AN1.png")), frame_with(crop_of("AP1.png"))
post("/api/reset")
t0, log = time.time(), []
for label, frame, seconds in (("A (AN01P9687)", A, 4), ("B (AP05BY7799)", B, 8)):
    end = time.time() + seconds
    while time.time() < end:
        r = post("/api/camera-frame", frame)
        log.append((round(time.time() - t0, 1), label, r.get("reading"), r.get("logged")))
        time.sleep(0.3)
print("t(s)  showing          reading        logged")
last = None
for t, label, reading, logged in log:
    if (label, reading, logged) != last or logged:
        print(f"{t:5}  {label:15} {str(reading):13}  {logged or ''}")
    last = (label, reading, logged)
rows = [r["plate"] for r in json.load(urllib.request.urlopen(BASE + "/api/plates?limit=10"))]
print("\nrecorded plates, newest first:", rows)
post("/api/reset")
