"""How much CPU does the running app use per camera frame?

usage: python measure_cpu.py [frames]
Sends the same webcam-style frame to /api/camera-frame (a plate in view) and a blank frame (nothing in view),
and reports wall time, CPU-seconds used by the server process, and the average number of cores busy.
"""
import json
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import psutil

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE = "http://127.0.0.1:8010"
SCRATCH = Path(r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin\c44b140b-18e7-43ee-a3e1-b7d47d376429\scratchpad")


def post(path, data=None, name="f.jpg"):
    if data is None:
        req = urllib.request.Request(BASE + path, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    else:
        b = uuid.uuid4().hex
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def server_process():
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "uvicorn" in c and "app.server:app" in c and "8010" in c:
            kids = p.children(recursive=True)
            return max([p] + kids, key=lambda q: q.cpu_times().user + q.cpu_times().system)
    raise SystemExit("app is not running on 8010")


def jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


plate_frame = jpeg(cv2.imread(str(SCRATCH / "fake_cam_frame.png")))
blank_frame = jpeg(np.full((720, 1280, 3), 60, np.uint8))
proc = server_process()
post("/api/reset"); post("/api/camera-frame", plate_frame); post("/api/camera-frame", plate_frame)  # warm-up
print(f"server PID {proc.pid}, {psutil.cpu_count()} logical cores, {N} frames each\n")
print(f"{'situation':34} {'ms/frame':>9} {'CPU-s/frame':>12} {'cores busy':>11}")
for name, frame in (("plate in view (already recorded)", plate_frame), ("nothing in view", blank_frame)):
    c0, t0 = proc.cpu_times(), time.time()
    for _ in range(N):
        post("/api/camera-frame", frame)
    c1, wall = proc.cpu_times(), time.time() - t0
    cpu = (c1.user + c1.system) - (c0.user + c0.system)
    print(f"{name:34} {wall / N * 1000:9.0f} {cpu / N:12.2f} {cpu / wall:11.1f}", flush=True)
post("/api/reset")
