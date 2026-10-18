"""Local web app: python -m uvicorn app.server:app --port 8000 (run from the project folder)."""
import os

# Limit the math libraries' thread pools before they load (setting it later does not reach them). Without this the
# detector spreads across nearly every core, and the whole PC feels stuck while the camera is on.
_THREADS = str(max(2, (os.cpu_count() or 4) // 2))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)
import asyncio
import hashlib
import hmac
import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engine import PPE_ITEMS, SNAP_DIR, engine

HERE = Path(__file__).parent
UPLOADS = HERE / "uploads"
UPLOADS.mkdir(exist_ok=True)

app = FastAPI(title="PPE ANPR")


@app.on_event("startup")
def warm_up():
    """Load the plate detector and OCR and run them once in the background, so the first photo or camera frame
    is not charged for model start-up (several seconds)."""
    def work():
        try:
            import numpy as np
            det, ocr = engine._photo_models()
            det.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, device="cpu", verbose=False)
            ocr.read(np.full((80, 400, 3), 255, np.uint8))
        except Exception:  # warm-up is only an optimisation; a failure must not stop the app
            pass
    threading.Thread(target=work, daemon=True).start()
app.mount("/snaps", StaticFiles(directory=SNAP_DIR), name="snaps")


# ---- live events: every recorded plate is pushed to WebSocket clients and to registered webhooks ----
WEBHOOKS_PATH = HERE / "webhooks.json"
RETRY_WAITS = (1, 3, 9)       # seconds to wait between delivery attempts of one event
WEBHOOK_TIMEOUT = 5           # seconds to wait for a receiver to answer
ws_clients: set = set()
delivery_log: deque = deque(maxlen=30)  # newest last: what happened to the latest deliveries
_events: "asyncio.Queue" = None
_tasks: set = set()           # keeps delivery tasks alive until they finish


def load_hooks():
    try:
        return json.loads(WEBHOOKS_PATH.read_text())
    except (OSError, ValueError):
        return []


def save_hooks(hooks):
    WEBHOOKS_PATH.write_text(json.dumps(hooks, indent=2))


async def deliver(hook, event, attempts=len(RETRY_WAITS)):
    """POST the event to one webhook; retry with backoff. Signed with HMAC-SHA256 when the hook has a secret."""
    body = json.dumps(event).encode()
    headers = {"Content-Type": "application/json", "X-Event": event["event"]}
    if hook.get("secret"):
        headers["X-Signature"] = "sha256=" + hmac.new(hook["secret"].encode(), body, hashlib.sha256).hexdigest()
    result = {"ok": False, "error": "no attempt"}
    async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
        for i in range(attempts):
            try:
                r = await client.post(hook["url"], content=body, headers=headers)
                result = {"ok": r.status_code < 300, "status": r.status_code, "error": None if r.status_code < 300 else f"HTTP {r.status_code}"}
            except Exception as e:
                result = {"ok": False, "status": None, "error": type(e).__name__}
            if result["ok"]:
                break
            if i < attempts - 1:
                await asyncio.sleep(RETRY_WAITS[i])
    result.update(hook=hook["url"], event=event["event"], attempts=i + 1, at=time.strftime("%H:%M:%S"))
    delivery_log.append(result)
    return result


async def dispatcher():
    while True:
        event = await _events.get()
        for ws in list(ws_clients):
            try:
                await ws.send_json(event)
            except Exception:
                ws_clients.discard(ws)
        for hook in load_hooks():
            t = asyncio.create_task(deliver(hook, event))
            _tasks.add(t)
            t.add_done_callback(_tasks.discard)


@app.on_event("startup")
async def start_events():
    global _events
    _events = asyncio.Queue()
    loop = asyncio.get_running_loop()
    # the detector runs in worker threads, so hand its events over to the event loop safely
    engine.emit = lambda event: loop.call_soon_threadsafe(_events.put_nowait, event)
    t = asyncio.create_task(dispatcher())
    _tasks.add(t)


@app.websocket("/ws")
async def websocket_events(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    await ws.send_json({"event": "hello", "message": "connected; plate.read events will arrive here"})
    try:
        while True:
            await ws.receive_text()  # clients send nothing; this just notices when they disconnect
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


class HookReq(BaseModel):
    url: str
    secret: str = ""


def _public(h):
    return {"id": h["id"], "url": h["url"], "has_secret": bool(h.get("secret"))}


@app.get("/api/webhooks")
def list_webhooks():
    return {"hooks": [_public(h) for h in load_hooks()], "log": list(delivery_log)[::-1],
            "websocket_clients": len(ws_clients)}


@app.post("/api/webhooks")
def add_webhook(r: HookReq):
    u = urlparse(r.url.strip())
    if u.scheme not in ("http", "https") or not u.netloc:
        raise HTTPException(400, "the webhook address must start with http:// or https://")
    hooks = load_hooks()
    hook = {"id": uuid.uuid4().hex[:8], "url": r.url.strip(), "secret": r.secret}
    hooks.append(hook)
    save_hooks(hooks)
    return _public(hook)


@app.delete("/api/webhooks/{hook_id}")
def delete_webhook(hook_id: str):
    hooks = load_hooks()
    if not any(h["id"] == hook_id for h in hooks):
        raise HTTPException(404, "no such webhook")
    save_hooks([h for h in hooks if h["id"] != hook_id])
    return {"ok": True}


@app.post("/api/webhooks/{hook_id}/test")
async def test_webhook(hook_id: str):
    hook = next((h for h in load_hooks() if h["id"] == hook_id), None)
    if not hook:
        raise HTTPException(404, "no such webhook")
    return await deliver(hook, {"event": "webhook.test", "message": "test from Number Plate Reader",
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, attempts=1)


class StartReq(BaseModel):
    source: str
    every: int = 3
    required: list[str] = ["helmet", "vest"]
    mode: str = "gate"  # gate | full


class ZoneReq(BaseModel):
    # fractions of the frame (0..1); all four None clears the box so the whole frame is the gate
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    return engine.status


@app.post("/api/start")
def start(r: StartReq):
    engine.start(r.source.strip().strip('"'), r.every, r.required, r.mode)
    return engine.status


@app.post("/api/stop")
def stop():
    engine.stop()
    return engine.status


@app.post("/api/zone")
def set_zone(z: ZoneReq):
    vals = (z.x1, z.y1, z.x2, z.y2)
    try:
        engine.set_zone(None if all(v is None for v in vals) else vals)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    return {"zone": engine.zone}


@app.post("/api/reset")
def reset():
    engine.reset()
    return {"ok": True}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    dest = UPLOADS / Path(file.filename).name
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return {"path": str(dest)}


@app.post("/api/read-photo")
def read_photo(file: UploadFile = File(...), save: int = 1, log: str = ""):
    """Read plates in one image. save=0 skips writing the annotated image; log=photo|camera also records it."""
    try:
        return engine.read_photo(file.file.read(), save_image=bool(save), log_source=log if log in ("photo", "camera") else "")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/camera-detect")
def camera_detect(file: UploadFile = File(...)):
    """One webcam frame, fast: just where the plates are, for the live box on the video."""
    try:
        return engine.camera_detect(file.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/camera-frame")
def camera_frame(file: UploadFile = File(...)):
    """One webcam frame: returns the reading, and logged=<plate> when a plate has just been confirmed and recorded."""
    try:
        return engine.camera_read(file.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/summary")
def summary():
    p = engine.q("SELECT COUNT(*) n, SUM(compliant) ok, SUM(1-helmet) nh, SUM(1-vest) nv, SUM(1-goggles) ng, "
                 "SUM(1-safety_shoe) ns FROM people")[0]
    pl = engine.q("SELECT COUNT(*) n, SUM(valid) ok, SUM(status='ok') good, SUM(status='review') review, "
                  "SUM(status='no_plate') noplate FROM plates")[0]
    by_hour = engine.q("SELECT substr(ts,12,2) h, COUNT(*) n FROM plates GROUP BY h ORDER BY h")
    n = p["n"] or 0
    ok = p["ok"] or 0
    return {
        "people": n, "compliant": ok, "violations": n - ok,
        "missed": {"helmet": p["nh"] or 0, "vest": p["nv"] or 0, "goggles": p["ng"] or 0, "safety_shoe": p["ns"] or 0},
        "plates": pl["n"] or 0, "plates_valid": pl["ok"] or 0, "plates_ok": pl["good"] or 0,
        "plates_review": (pl["review"] or 0) + (pl["noplate"] or 0), "plates_by_hour": by_hour,
        "items": PPE_ITEMS,
    }


@app.get("/api/violations")
def violations(limit: int = 30):
    return engine.q("SELECT * FROM people WHERE compliant=0 ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/api/plates")
def plates(limit: int = 30):
    return engine.q("SELECT * FROM plates ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/stream")
def stream():
    def gen():
        last = None
        while True:
            jpg = engine.jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
