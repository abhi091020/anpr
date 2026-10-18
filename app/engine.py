"""Detection engine. Two modes on one stream, results into SQLite:

  gate  - only vehicles inside a user-drawn box are looked at; one plate record per vehicle (default)
  full  - whole frame: people + PPE + plates (the earlier prototype behaviour)
"""
import difflib
import json
import os
import sqlite3
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plate_ocr import IN_PLATE, PlateReader, clean_plate  # noqa: E402

# The detector and the OCR would otherwise use nearly every core, which makes the whole PC feel stuck while the
# camera runs. Half the logical cores (the physical ones) keeps almost all of the speed and leaves the rest free.
CPU_THREADS = max(2, (os.cpu_count() or 4) // 2)
torch.set_num_threads(CPU_THREADS)
cv2.setNumThreads(2)  # image decoding and resizing need few threads

DB_PATH = ROOT / "app" / "data.db"
ZONE_PATH = ROOT / "app" / "zone.json"
SNAP_DIR = ROOT / "app" / "snaps"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

PPE_ITEMS = ["helmet", "vest", "goggles", "safety_shoe"]
PPE_NAMES = {"Gloves": "gloves", "Vest": "vest", "goggles": "goggles", "helmet": "helmet",
             "mask": "mask", "safety_shoe": "safety_shoe"}

MIN_FRAMES = 10        # full mode: inference frames a person must be tracked before we judge them
WORN_RATIO = 0.25      # full mode: item counts as worn if seen on this share of those frames
MAX_OCR_READS = 8      # full mode: OCR reads per tracked plate

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
ZONE_FRACTION = 0.5    # gate mode: a vehicle counts as "in the box" when this share of the smaller of
                       # (vehicle, box) lies inside the other
LEAVE_AFTER = 4        # gate mode: inference frames outside the box before a vehicle is finalised
GATE_MAX_READS = 6     # gate mode: OCR reads per vehicle; the vehicle is logged once it has this many
MIN_GATE_FRAMES = 3    # gate mode: ignore vehicles that only brush the box
MIN_VEHICLE_AREA = 0.08  # gate mode: a vehicle must cover this share of the box, so far-away cars are ignored
NO_PLATE_MIN_FRAMES = 8  # gate mode: a "no plate read" record needs the vehicle to stay in the box this long
MERGE_SECONDS = 15     # gate mode: the same plate text within this many seconds is the same vehicle
OK_CONFIDENCE = 0.85   # photo/camera: OCR confidence at or above this is "ok", below is flagged for a look
PHOTO_MAX_IMGSZ = 1920  # photo: detector size is raised up to this for large photos so small plates are not lost
CAMERA_IMGSZ = 960     # camera: second, larger detector size, tried only when the native 640 finds no plate (far or small plates)
NATIVE_IMGSZ = 640     # the size the plate detector was trained at: its boxes are most accurate there, and larger sizes can cut a
                       # big plate short (measured: a plate filling the image was boxed at 100% of its width at 640, 75% at 960)
CROP_PAD = 0.05        # grow each plate box by this share on every side before reading, so edge characters are not clipped
CAMERA_AGREE = 2       # camera: this many of the last CAMERA_WINDOW frames must give the same plate text ...
CAMERA_WINDOW = 6
CAMERA_SURE = 0.97     # ... unless the OCR confidence is at least this, which confirms a plate on the first frame
CAMERA_COOLDOWN = 15   # camera: seconds before the same plate can be recorded again
CAMERA_RECHECK_S = 1.5  # camera: while a recorded plate seems to still be in view, read it again this often to notice a swap
SAME_PLATE = 0.85      # camera: two readings at least this similar (0..1) are the same plate, so OCR wobble is not a new plate
OK_VOTES = 6           # gate mode: a plate is "ok" when a valid-format text got this many votes
                       # (a valid read is worth 3, so two agreeing valid reads)

SCHEMA = """
CREATE TABLE IF NOT EXISTS people(id INTEGER PRIMARY KEY, ts TEXT, track INTEGER, helmet INT, vest INT,
  goggles INT, safety_shoe INT, missing TEXT, compliant INT, snapshot TEXT);
CREATE TABLE IF NOT EXISTS plates(id INTEGER PRIMARY KEY, ts TEXT, track INTEGER, plate TEXT, valid INT,
  reads INT, snapshot TEXT, vehicle_snap TEXT, status TEXT);
"""


def db():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    cols = {r["name"] for r in c.execute("PRAGMA table_info(plates)")}
    for col in ("vehicle_snap", "status", "source"):  # older databases lack these
        if col not in cols:
            c.execute(f"ALTER TABLE plates ADD COLUMN {col} TEXT")
    c.commit()
    return c


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.conn = db()
        self.models = None
        self.thread = None
        self.stop_flag = threading.Event()
        self.jpeg = None
        self.mode = "gate"
        self.recent = {}  # plate text -> (db row, last seen time), merges track-id fragments (full mode)
        self.gate_recent = []  # recent gate records {row, t, box, text, votes}, merges track-id fragments (gate mode)
        self.vclock = 0.0
        self.emit = lambda event: None  # set by the web server; called with one dict each time a plate is recorded
        self.det_lock = threading.Lock()  # the camera's fast box loop and its slower reading loop share one detector
        self.cam_reads = deque(maxlen=CAMERA_WINDOW)  # camera: recent plate texts, for confirming a plate
        self.cam_logged = {}  # camera: plate text -> time it was recorded
        self.cam_last = None  # camera: the plate just recorded {box, text, confidence, t}, so it is not read again while it stays
        self.zone = self._load_zone()  # (x1, y1, x2, y2) as 0..1 fractions of the frame, or None = whole frame
        self.status = {"running": False, "source": None, "frames": 0, "fps": 0.0, "error": None,
                       "message": "", "mode": "gate", "zone": self.zone, "in_zone": 0}

    # ---- gate box ----
    def _load_zone(self):
        try:
            z = json.loads(ZONE_PATH.read_text())
            return tuple(z) if z else None
        except (OSError, ValueError):
            return None

    def set_zone(self, zone):
        if zone is not None:
            x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in zone)
            zone = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if zone[2] - zone[0] < 0.02 or zone[3] - zone[1] < 0.02:
                raise ValueError("gate box is too small")
        self.zone = zone
        self.status["zone"] = zone
        ZONE_PATH.write_text(json.dumps(zone))

    # ---- lifecycle ----
    def _load(self):
        if self.models is None:
            self.status["message"] = "loading models"
            self.models = {
                "person": YOLO("yolov8n.pt"),  # also the vehicle detector in gate mode (different classes)
                "ppe": YOLO(str(ROOT / "models/ppe/best.pt")),
                "plate": YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt")),
                "ocr": PlateReader(),
            }

    def start(self, source, every=3, required=None, mode="gate"):
        self.stop()
        self.stop_flag.clear()
        self.gate_recent = []  # the video clock restarts at 0, so entries from an earlier run must not linger
        self.vclock = 0.0
        self.mode = mode if mode in ("gate", "full") else "gate"
        self.required = [r for r in (required or ["helmet", "vest"]) if r in PPE_ITEMS]
        self.every = max(1, int(every))
        self.status.update(running=True, source=str(source), frames=0, fps=0.0, error=None, message="starting",
                           mode=self.mode, zone=self.zone, in_zone=0)
        self.thread = threading.Thread(target=self._loop, args=(source,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=15)
        self.status["running"] = False

    def reset(self):
        with self.lock:
            self.conn.executescript("DELETE FROM people; DELETE FROM plates;")
            self.conn.commit()
        self.recent.clear()
        self.gate_recent = []
        self.cam_logged.clear()  # after clearing the history, the same plate must be recordable again
        self.cam_last = None
        self.cam_reads.clear()
        for p in SNAP_DIR.glob("*.jpg"):
            p.unlink(missing_ok=True)

    # ---- main loop ----
    def _loop(self, source):
        veh, people, plates = {}, {}, {}
        frame = None
        try:
            self._load()
            src = int(source) if str(source).isdigit() else str(source)
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open source: {source}")
            is_file = Path(str(source)).exists()
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            for m in ("person", "plate"):  # drop old predictors so tracker ids start fresh for each run
                self.models[m].predictor = None
            last = {"persons": [], "ppe": [], "plates": [], "veh": []}
            n, t0 = 0, time.time()
            self.status["message"] = "running"
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.status["message"] = "stream ended"
                    break
                n += 1
                self.vclock = n / src_fps  # video time in seconds: the right clock for merging records when a file is
                #                            processed slower than real time (live streams give the same value)
                if n % self.every == 0:
                    last = self._infer_gate(frame, veh) if self.mode == "gate" else self._infer(frame, people, plates)
                self._draw_gate(frame, last) if self.mode == "gate" else self._draw(frame, last)
                good, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if good:
                    self.jpeg = buf.tobytes()
                self.status["frames"] = n
                self.status["fps"] = round(n / max(time.time() - t0, 1e-6), 1)
                if is_file and self.status["fps"] > src_fps * 1.5:
                    time.sleep(0.5 / src_fps)  # don't race through a file far faster than real time
            cap.release()
            if self.mode == "gate":  # a vehicle still in the box when the stream ends still gets logged
                for tid, st in veh.items():
                    if not st["logged"] and st["frames"] >= MIN_GATE_FRAMES:
                        self._finalize_vehicle(tid, st)
        except Exception as e:  # surfaced to the UI
            self.status["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.status["running"] = False

    # ---- gate mode ----
    def _zone_px(self, w, h):
        z = self.zone or (0.0, 0.0, 1.0, 1.0)
        return int(z[0] * w), int(z[1] * h), int(z[2] * w), int(z[3] * h)

    @staticmethod
    def _overlap(a, b):
        """Share of the smaller box that lies inside the other."""
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return (ix * iy) / small if small > 0 else 0.0

    def _infer_gate(self, frame, veh):
        m = self.models
        h, w = frame.shape[:2]
        zone = self._zone_px(w, h)
        vr = m["person"].track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.35, imgsz=640,
                               tracker="bytetrack.yaml", device="cpu", verbose=False)[0]
        seen_in_zone, shown = {}, []
        zone_area = max((zone[2] - zone[0]) * (zone[3] - zone[1]), 1)
        if vr.boxes is not None and vr.boxes.id is not None:
            for b, tid in zip(vr.boxes.xyxy.tolist(), vr.boxes.id.int().tolist()):
                box = tuple(map(int, b))
                big_enough = (box[2] - box[0]) * (box[3] - box[1]) >= MIN_VEHICLE_AREA * zone_area
                inside = big_enough and self._overlap(box, zone) >= ZONE_FRACTION
                shown.append((tid, box, inside))
                if inside:
                    seen_in_zone[tid] = box

        for tid in seen_in_zone:
            veh.setdefault(tid, {"frames": 0, "miss": 0, "votes": Counter(), "reads": 0, "logged": False,
                                 "plate_conf": 0.0, "plate_crop": None, "veh_crop": None})
        for tid, st in veh.items():
            if tid in seen_in_zone:
                st["frames"] += 1
                st["miss"] = 0
                st["box"] = seen_in_zone[tid]
            elif not st["logged"]:
                st["miss"] += 1
                if st["miss"] >= LEAVE_AFTER and st["frames"] >= MIN_GATE_FRAMES:
                    self._finalize_vehicle(tid, st)
                elif st["miss"] >= LEAVE_AFTER:
                    st["logged"] = True  # only brushed the box: drop it without a record

        plate_out = []
        if seen_in_zone:  # plate detection runs only on the box, and only while a vehicle is in it
            zx1, zy1, zx2, zy2 = zone
            pr = m["plate"].predict(frame[zy1:zy2, zx1:zx2], imgsz=640, conf=0.4, device="cpu", verbose=False)[0]
            for b in pr.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                x1, x2, y1, y2 = x1 + zx1, x2 + zx1, y1 + zy1, y2 + zy1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                owners = [(t, bx) for t, bx in seen_in_zone.items() if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]]
                if not owners:
                    continue
                tid, vbox = min(owners, key=lambda o: (o[1][2] - o[1][0]) * (o[1][3] - o[1][1]))
                st = veh[tid]
                text = ""
                if st["reads"] < GATE_MAX_READS and not st["logged"]:
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    raw, _ = m["ocr"].read(crop)
                    text, valid = clean_plate(raw)
                    if text:
                        st["votes"][text] += 1 + (2 if valid else 0)
                        st["reads"] += 1
                    if float(b.conf) > st["plate_conf"] and crop.size:
                        st["plate_conf"], st["plate_crop"] = float(b.conf), crop.copy()
                        st["veh_crop"] = frame[max(vbox[1], 0):vbox[3], max(vbox[0], 0):vbox[2]].copy()
                    if st["reads"] >= GATE_MAX_READS:
                        self._finalize_vehicle(tid, st)
                best = st["votes"].most_common(1)[0][0] if st["votes"] else text
                plate_out.append((best, bool(IN_PLATE.match(best)), x1, y1, x2, y2))
        self.status["in_zone"] = len(seen_in_zone)
        return {"zone": zone, "veh": shown, "plates": plate_out, "persons": [], "ppe": []}

    def _finalize_vehicle(self, tid, st):
        """Write one record for a vehicle that was in the gate box, then mark it done."""
        st["logged"] = True
        if st["votes"]:
            text, votes = st["votes"].most_common(1)[0]
            valid = bool(IN_PLATE.match(text))
            status = "ok" if valid and votes >= OK_VOTES else "review"
        elif st["frames"] < NO_PLATE_MIN_FRAMES:
            return  # too brief to be a vehicle that stopped at the gate: drop without a record
        else:
            text, valid, status = "", False, "no_plate"  # a vehicle was in the box but no plate could be read
        now = self.vclock
        votes_top = st["votes"].most_common(1)[0][1] if st["votes"] else 0
        box = st.get("box")
        # the tracker often splits one vehicle into several ids: a record whose vehicle box overlaps one from the
        # last few seconds (or that has the same text) is the same vehicle, and the better-voted reading wins
        same = next((g for g in self.gate_recent if now - g["t"] < MERGE_SECONDS and
                     ((box and g["box"] and self._overlap(box, g["box"]) >= 0.5) or (text and text == g["text"]))), None)
        if same:
            same["t"], same["box"] = now, box or same["box"]
            with self.lock:
                if votes_top > same["votes"] or (not same["text"] and text):
                    snaps = (None, None)
                    if st["plate_crop"] is not None:
                        stamp = int(time.time() * 1000)
                        snaps = (f"c{stamp}_{tid}.jpg", f"v{stamp}_{tid}.jpg")
                        cv2.imwrite(str(SNAP_DIR / snaps[0]), st["plate_crop"])
                        if st["veh_crop"] is not None and st["veh_crop"].size:
                            cv2.imwrite(str(SNAP_DIR / snaps[1]), st["veh_crop"])
                    self.conn.execute(
                        "UPDATE plates SET plate=?, valid=?, status=?, reads=reads+?, snapshot=COALESCE(?,snapshot), "
                        "vehicle_snap=COALESCE(?,vehicle_snap) WHERE id=?",
                        (text, int(valid), status, st["reads"], snaps[0], snaps[1], same["row"]))
                    same["text"], same["votes"] = text, votes_top
                else:
                    self.conn.execute("UPDATE plates SET reads=reads+? WHERE id=?", (st["reads"], same["row"]))
                self.conn.commit()
            st["plate_crop"] = st["veh_crop"] = None
            return
        stamp = int(time.time() * 1000)
        plate_snap = veh_snap = None
        if st["plate_crop"] is not None:
            plate_snap = f"c{stamp}_{tid}.jpg"
            cv2.imwrite(str(SNAP_DIR / plate_snap), st["plate_crop"])
        if st["veh_crop"] is not None and st["veh_crop"].size:
            veh_snap = f"v{stamp}_{tid}.jpg"
            cv2.imwrite(str(SNAP_DIR / veh_snap), st["veh_crop"])
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO plates(ts,track,plate,valid,reads,snapshot,vehicle_snap,status) VALUES(?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), tid, text, int(valid), st["reads"], plate_snap,
                 veh_snap, status))
            self.conn.commit()
        self._announce(cur.lastrowid, datetime.now().isoformat(timespec="seconds"), text, None, status, "gate",
                       plate_snap or veh_snap)
        self.gate_recent.append({"row": cur.lastrowid, "t": now, "box": box, "text": text, "votes": votes_top})
        self.gate_recent = [g for g in self.gate_recent if now - g["t"] < 60]
        st["plate_crop"] = st["veh_crop"] = None

    def _draw_gate(self, frame, last):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._zone_px(w, h)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(frame, "GATE", (x1 + 6, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        for tid, (a, b, c, d), inside in last.get("veh", []):
            color = (0, 200, 0) if inside else (140, 140, 140)
            cv2.rectangle(frame, (a, b), (c, d), color, 2 if inside else 1)
            cv2.putText(frame, "in gate" if inside else "outside", (a, max(b - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for text, valid, a, b, c, d in last.get("plates", []):
            cv2.rectangle(frame, (a, b), (c, d), (255, 160, 0), 2)
            cv2.putText(frame, (text or "plate") + ("" if valid else " ?"), (a, max(b - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2)

    # ---- full mode (whole frame: people + PPE + plates) ----
    def _infer(self, frame, people, plates):
        m = self.models
        pr = m["person"].track(frame, persist=True, classes=[0], conf=0.4, imgsz=640,
                               tracker="bytetrack.yaml", device="cpu", verbose=False)[0]
        persons = []
        if pr.boxes is not None and pr.boxes.id is not None:
            for b, tid in zip(pr.boxes.xyxy.tolist(), pr.boxes.id.int().tolist()):
                persons.append((tid, *map(int, b)))
        dets = []
        for b in m["ppe"].predict(frame, imgsz=640, conf=0.35, device="cpu", verbose=False)[0].boxes:
            name = PPE_NAMES.get(m["ppe"].names[int(b.cls)])
            if name:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                dets.append((name, float(b.conf), x1, y1, x2, y2))

        for tid, x1, y1, x2, y2 in persons:
            st = people.setdefault(tid, {"frames": 0, "seen": Counter(), "done": False})
            st["frames"] += 1
            pw, ph = x2 - x1, y2 - y1
            seen_now = set()
            for name, _c, a1, b1, a2, b2 in dets:
                cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
                if x1 - 0.1 * pw <= cx <= x2 + 0.1 * pw and y1 - 0.1 * ph <= cy <= y2 + 0.1 * ph:
                    seen_now.add(name)
            st["seen"].update(seen_now)
            if not st["done"] and st["frames"] >= MIN_FRAMES:
                st["done"] = True
                self._record_person(frame, tid, st, (x1, y1, x2, y2))

        pl = m["plate"].track(frame, persist=True, conf=0.4, imgsz=640, tracker="bytetrack.yaml",
                              device="cpu", verbose=False)[0]
        plate_out = []
        if pl.boxes is not None and pl.boxes.id is not None:
            for b, tid in zip(pl.boxes.xyxy.tolist(), pl.boxes.id.int().tolist()):
                x1, y1, x2, y2 = map(int, b)
                st = plates.setdefault(tid, {"votes": Counter(), "reads": 0, "row": None})
                if st["reads"] < MAX_OCR_READS:
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    raw, _conf = m["ocr"].read(crop)
                    text, valid = clean_plate(raw)
                    if text:
                        st["votes"][text] += 1 + (2 if valid else 0)
                        st["reads"] += 1
                        if st["reads"] >= 2:
                            self._save_plate(tid, st, crop)
                best = st["votes"].most_common(1)[0][0] if st["votes"] else ""
                plate_out.append((best, bool(IN_PLATE.match(best)), x1, y1, x2, y2))
        return {"persons": persons, "ppe": dets, "plates": plate_out, "veh": [],
                "state": {t: {"done": s["done"], "seen": set(s["seen"])} for t, s in people.items()}}

    def _record_person(self, frame, tid, st, box):
        ratio = {k: st["seen"][k] / st["frames"] for k in PPE_ITEMS}
        worn = {k: int(ratio[k] >= WORN_RATIO) for k in PPE_ITEMS}
        missing = [k for k in self.required if not worn[k]]
        x1, y1, x2, y2 = box
        snap = f"p{int(time.time() * 1000)}_{tid}.jpg"
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size:
            cv2.imwrite(str(SNAP_DIR / snap), crop)
        with self.lock:
            self.conn.execute(
                "INSERT INTO people(ts,track,helmet,vest,goggles,safety_shoe,missing,compliant,snapshot) VALUES(?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), tid, worn["helmet"], worn["vest"], worn["goggles"],
                 worn["safety_shoe"], ",".join(missing), int(not missing), snap))
            self.conn.commit()

    def _save_plate(self, tid, st, crop):
        text = st["votes"].most_common(1)[0][0]
        valid = int(bool(IN_PLATE.match(text)))
        now = time.time()
        with self.lock:
            prev = self.recent.get(text)
            if st["row"] is None and prev and now - prev[1] < 15:
                st["row"] = prev[0]  # same text just seen under another track id: same vehicle
            if st["row"] is not None:
                self.recent[text] = (st["row"], now)
            if st["row"] is None:
                snap = f"c{int(time.time() * 1000)}_{tid}.jpg"
                if crop.size:
                    cv2.imwrite(str(SNAP_DIR / snap), crop)
                cur = self.conn.execute(
                    "INSERT INTO plates(ts,track,plate,valid,reads,snapshot,status) VALUES(?,?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), tid, text, valid, st["reads"], snap,
                     "ok" if valid else "review"))
                st["row"] = cur.lastrowid
                self.recent[text] = (st["row"], now)
            else:
                self.conn.execute("UPDATE plates SET plate=?, valid=?, reads=?, status=? WHERE id=?",
                                  (text, valid, st["reads"], "ok" if valid else "review", st["row"]))
            self.conn.commit()

    def _draw(self, frame, last):
        state = last.get("state", {})
        for tid, x1, y1, x2, y2 in last["persons"]:
            s = state.get(tid, {})
            if not s.get("done"):
                color, label = (160, 160, 160), f"person {tid} (checking)"
            else:
                miss = [k for k in self.required if k not in s["seen"]]
                color = (0, 0, 220) if miss else (0, 180, 0)
                label = f"person {tid}: " + ("missing " + ",".join(miss) if miss else "OK")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for name, conf, x1, y1, x2, y2 in last["ppe"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(frame, name, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        for text, valid, x1, y1, x2, y2 in last["plates"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
            cv2.putText(frame, (text or "plate") + ("" if valid else " ?"), (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2)

    # ---- single photo / webcam frame (no gate box, no video) ----
    def _photo_models(self):
        """The plate detector and the OCR, loaded on first use. Reuses the video pipeline's OCR when it is loaded."""
        if getattr(self, "photo_plate", None) is None:
            self.photo_plate = YOLO(str(ROOT / "models/plate/license-plate-finetune-v1n.pt"))
        if self.models is not None:
            return self.photo_plate, self.models["ocr"]
        if getattr(self, "photo_ocr", None) is None:
            self.photo_ocr = PlateReader()
        return self.photo_plate, self.photo_ocr

    @staticmethod
    def _decode(data):
        import numpy as np
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("that file is not an image")
        return img

    def _find_plates(self, img, sizes):
        """Detect plates at one or more detector sizes and merge boxes that mark the same plate (their union).

        Small plates in big images need a large detector size; large plates need the native 640, because bigger sizes
        draw a too-small box around them. Running both and taking the union gives a box that fits either case.
        Returns [[x1, y1, x2, y2, confidence], ...], most confident first."""
        det, _ = self._photo_models()
        found = []
        for sz in dict.fromkeys(sizes):
            with self.det_lock:
                res = det.predict(img, imgsz=sz, conf=0.25, device="cpu", verbose=False)[0]
            found += [[*map(float, b.xyxy[0]), float(b.conf)] for b in res.boxes]
        merged = []
        for f in sorted(found, key=lambda f: -f[4]):
            for m in merged:
                if self._overlap(tuple(f[:4]), tuple(m[:4])) >= 0.5:
                    m[0], m[1], m[2], m[3] = min(m[0], f[0]), min(m[1], f[1]), max(m[2], f[2]), max(m[3], f[3])
                    break
            else:
                merged.append(f[:])
        return merged

    def _read_boxes(self, img, boxes, mode="auto"):
        """Read the plate inside each box (with a little padding). Returns a list of plate dicts."""
        _, ocr = self._photo_models()
        h, w = img.shape[:2]
        out = []
        for x1, y1, x2, y2, conf in boxes:
            px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
            cx1, cy1, cx2, cy2 = max(int(x1 - px), 0), max(int(y1 - py), 0), min(int(x2 + px), w), min(int(y2 + py), h)
            raw, c = ocr.read(img[cy1:cy2, cx1:cx2], mode=mode)
            text, valid = clean_plate(raw)
            out.append({"plate": text, "raw": raw, "valid": valid, "confidence": round(c, 2),
                        "detector_conf": round(conf, 2), "box": [int(x1), int(y1), int(x2), int(y2)]})
        return out

    def _detect_read(self, img, imgsz, max_plates=5):
        """Find the plates in the image (the most confident max_plates) and read them. Returns (plates, timing in ms)."""
        t0 = time.time()
        boxes = self._find_plates(img, imgsz if isinstance(imgsz, (list, tuple)) else [imgsz])[:max_plates]
        t1 = time.time()
        out = self._read_boxes(img, boxes, mode="best")  # both readers, keep the better (measured: 92% vs 86% on easy photos)
        return out, {"detect_ms": round((t1 - t0) * 1000), "ocr_ms": round((time.time() - t1) * 1000)}

    def _save_and_log(self, img, out, source=""):
        """Write the annotated image; with a source, also record the best plate. Returns (image name, plate or None)."""
        for p in out:
            x1, y1, x2, y2 = p["box"]
            color = (0, 180, 0) if p["confidence"] >= OK_CONFIDENCE else (0, 160, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, img.shape[1] // 300))
            cv2.putText(img, p["plate"] or "?", (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.7, img.shape[1] / 900), color, 2)
        name = f"photo{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(SNAP_DIR / name), img)
        best = max((p for p in out if p["plate"]), key=lambda p: p["confidence"], default=None)
        if source and best:
            ts = datetime.now().isoformat(timespec="seconds")
            status = "ok" if best["confidence"] >= OK_CONFIDENCE else "review"
            with self.lock:
                cur = self.conn.execute(
                    "INSERT INTO plates(ts,plate,valid,reads,snapshot,status,source) VALUES(?,?,?,?,?,?,?)",
                    (ts, best["plate"], int(best["valid"]), 1, name, status, source))
                self.conn.commit()
            self._announce(cur.lastrowid, ts, best["plate"], best["confidence"], status, source, name)
        return name, (best["plate"] if source and best else None)

    def _announce(self, row_id, ts, plate, confidence, status, source, image):
        """Tell the web server a plate was recorded, so it can push it to WebSocket clients and webhooks."""
        try:
            self.emit({"event": "plate.read", "id": row_id, "plate": plate, "confidence": confidence,
                       "status": status, "source": source, "time": ts, "image": f"/snaps/{image}" if image else None})
        except Exception:  # a broken listener must never stop plate recording
            pass

    def read_photo(self, data, save_image=True, log_source=""):
        """Read the plates in one uploaded photo. Large photos are searched at a larger detector size, because a
        plate in a 4000-pixel photo shrinks to a few pixels when the whole image is squeezed to 640."""
        img = self._decode(data)
        h, w = img.shape[:2]
        big = int(min(PHOTO_MAX_IMGSZ, max(NATIVE_IMGSZ, -(-max(h, w) // 32) * 32)))
        sizes = [NATIVE_IMGSZ] if big <= NATIVE_IMGSZ else [NATIVE_IMGSZ, big]  # see _find_plates
        out, timing = self._detect_read(img, sizes)
        result = {"plates": out, "timing": timing, "detector_sizes": sizes}
        if save_image or (log_source and out):
            name, logged = self._save_and_log(img, out, log_source)
            result["image"] = name
            if logged:
                result["logged"] = logged
        return result

    def camera_detect(self, data):
        """Fast step for the live overlay: only find where the plates are (about 0.15 s), without reading them."""
        img = self._decode(data)
        det, _ = self._photo_models()
        t0 = time.time()
        with self.det_lock:
            res = det.predict(img, imgsz=CAMERA_IMGSZ, conf=0.25, device="cpu", verbose=False)[0]
        plates = [{"box": list(map(int, b.xyxy[0])), "detector_conf": round(float(b.conf), 2)}
                  for b in sorted(res.boxes, key=lambda b: -float(b.conf))[:3]]
        return {"plates": plates, "timing": {"detect_ms": round((time.time() - t0) * 1000)}}

    def camera_read(self, data):
        """Read one webcam frame and decide, here, whether a plate has been confirmed.

        Cheap when it can be: the detector runs on every frame, but the plate is read (the slow step) only when it is
        not the plate that was just recorded. A plate is confirmed when two of the last few frames agree, or at once
        when the OCR is very sure. It is recorded using this same frame, and not again while it stays in view."""
        img = self._decode(data)
        t0 = time.time()
        # the native size first: it boxes a plate held close to the camera accurately and is the fastest; the larger
        # size is tried only when that finds nothing (a small or far plate)
        boxes = self._find_plates(img, [NATIVE_IMGSZ]) or self._find_plates(img, [CAMERA_IMGSZ])
        detect_ms = round((time.time() - t0) * 1000)
        result = {"plates": [], "reading": None, "timing": {"detect_ms": detect_ms, "ocr_ms": 0}}
        if not boxes:
            self.cam_reads.clear()
            return result
        top = boxes[0]  # one plate is held up; reading more would only cost time
        box = list(map(int, top[:4]))
        now = time.time()
        last = self.cam_last
        same_place = bool(last and now - last["t"] < CAMERA_COOLDOWN and self._overlap(tuple(box), tuple(last["box"])) >= 0.6)
        if same_place and now - last["read_t"] < CAMERA_RECHECK_S:
            last["t"], last["box"] = now, box  # read a moment ago and still in the same place: no need to read it yet
            result["reading"] = last["text"]
            result["plates"] = [{"plate": last["text"], "raw": last["text"], "valid": bool(IN_PLATE.match(last["text"])),
                                 "confidence": last["confidence"], "detector_conf": round(top[4], 2), "box": box}]
            return result
        t1 = time.time()
        plate = self._read_boxes(img, [top], mode="fast")[0]
        result["timing"]["ocr_ms"] = round((time.time() - t1) * 1000)
        text = plate["plate"]
        result["plates"] = [plate]
        if same_place and text and difflib.SequenceMatcher(None, text, last["text"]).ratio() >= SAME_PLATE:
            # still the plate that was recorded (OCR can wobble by a character): keep it quiet and keep its reading
            last["t"], last["read_t"], last["box"] = now, now, box
            result["reading"] = last["text"]
            plate["plate"] = last["text"]
            return result
        if not text:
            self.cam_reads.clear()
            return result
        self.cam_reads.append(text)
        result["reading"] = text
        agreed = sum(t == text for t in self.cam_reads) >= CAMERA_AGREE or plate["confidence"] >= CAMERA_SURE
        seen = self.cam_logged.get(text)
        if seen is not None and now - seen < CAMERA_COOLDOWN:
            self.cam_logged[text] = now  # the same text again: keep it quiet until it has been gone a while
        elif agreed:
            self.cam_logged[text] = now
            self.cam_reads.clear()
            result["image"], result["logged"] = self._save_and_log(img, [plate], "camera")
            self.cam_last = {"box": box, "text": text, "confidence": plate["confidence"], "t": now, "read_t": now}
        return result

    # ---- queries ----
    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]


engine = Engine()
