# Number Plate Reader: production plan

Written 2026-10-18. This covers what exists, what was measured, and what to do to take it to a machine with a
powerful NVIDIA GPU and run it in production. Where something was **not tested**, it says so.

---

## 1. What exists today

A local web app that reads Indian number plates from **photos** and from a **live camera** (a laptop or phone camera
opened in the browser), records each plate, and pushes every recorded plate to a **WebSocket** and to **webhooks**.

- Runs on the CPU only (the development PC has no NVIDIA GPU).
- Dashboard: `http://127.0.0.1:8010` (port 8000 was already used by another program on the development PC).
- Start it with `start_app.bat`, or `python -m uvicorn app.server:app --host 127.0.0.1 --port 8010` from the project folder.

### How a plate is read

1. **Find the plate.** A YOLO plate detector draws a box. It is run at its native size (640) and, for big images, also at a
   larger size. The two boxes for the same plate are merged (their union), because larger sizes can cut a big plate short.
2. **Read the characters.** PaddleOCR reads the crop.
   - *Camera:* a fast recognition-only reader, about 0.3 s per frame.
   - *Photo:* the fast reader first; if its answer is a confident, valid plate it stops. Otherwise the slower full reader also
     runs and the better answer is kept. The slow reader is skipped on crops wider than 800 px (it takes 10 to 25 s there).
3. **Clean the text.** Junk read in front of the number (the "IND" mark on the blue strip) is dropped when what follows is a
   valid plate. **The end of a reading is never trimmed**, so a plate with an unusual format is shown whole.
4. **Record it.** In the camera, a plate is confirmed when two frames agree, or at once when the OCR is very sure. It is not
   recorded again while it stays in view, but is re-read every 1.5 s so a swapped plate is noticed.

### Files

| Path | What it is |
|---|---|
| `app/server.py` | Web server: dashboard, photo and camera endpoints, WebSocket, webhooks |
| `app/engine.py` | Detection, reading, confirming and recording plates; also a video/gate mode and a PPE mode that the clean dashboard does not use |
| `app/static/index.html` | The dashboard (Photo and Camera tabs, Live events, Recent reads) |
| `app/static/index_full.html` | The older full dashboard (video, gate box, PPE) kept for reference |
| `plate_ocr.py` | The OCR reader, the plate text cleanup, the best-of-both rule |
| `models/plate/` | Plate detector weights (YOLO, several sizes) |
| `models/ppe/` | PPE model (not used by the current dashboard) |
| `dataset-indian/` | 1,700 labelled Indian plate photos (CC BY 4.0) and `number_plate.xlsx` with the correct plates |
| `eval_plates.py` | Accuracy test over the labelled photos, with a fixed tune/test split |
| `bench_*.py`, `measure_cpu.py`, `test_*.py` | Speed/accuracy benchmarks and checks used while building |
| `make_ocr_dataset.py` | Cuts plate crops and labels for training (written, **not yet run**) |
| `app/data.db`, `app/webhooks.json`, `app/snaps/` | Runtime data: history, webhook settings **including secrets in plain text**, saved images |

### Interfaces

- `POST /api/read-photo?save=0|1&log=photo|camera`: read an uploaded image. `log` also records the best plate.
- `POST /api/camera-frame`: one webcam frame. The server confirms and records plates. Returns `reading` and `logged`.
- `POST /api/camera-detect`: where the plates are in a frame (fast; no reading).
- `GET /api/plates`, `GET /api/summary`, `POST /api/reset`: history.
- `GET/POST/DELETE /api/webhooks`, `POST /api/webhooks/{id}/test`: webhook management.
- `ws://host:8010/ws`: live events.

Event sent to the WebSocket and to every webhook (a webhook with a secret also carries
`X-Signature: sha256=<HMAC-SHA256 of the body>` and `X-Event: plate.read`):

```json
{"event": "plate.read", "id": 12, "plate": "AP39KL0404", "confidence": 1.0,
 "status": "ok", "source": "camera", "time": "2026-10-18T12:30:05", "image": "/snaps/photo1792281858523.jpg"}
```

Webhook delivery retries 3 times (after 1 s, 3 s, 9 s) and the dashboard shows the latest outcome.

---

## 2. What was measured (and what it means)

All numbers are from the labelled photos in `dataset-indian/`, on the development CPU. They are small samples, so treat
them as rough.

| Test | Result |
|---|---|
| Easy photos, 50 plates, current photo reader | **about 90% exact** |
| Large phone photos, 43 plates found of 50 | **about 49% exact** (the detector missed 7 of 50) |
| Webcam-size plates, 40 crops, fast reader vs the old reader | 70% vs 28% exact, about 5x faster |
| Camera frame, small plate | about 0.3 s; repeat frames of a recorded plate cost about 0 ms of reading |
| Full test on **all 1,696 photos**, before the later fixes | **about 40% exact** on held-out photos; 72% plate-found. Not rerun since. |

**Honest status:** the reader is good on clear, close plates and weak on large phone photos, tilted or embossed plates
(for example a bike plate where the "Q" was read as "0"), and anything blurred. It is **not yet production-grade**.
A plate-reading system should aim for 95% or better, with uncertain reads sent for human review.

**Not tested:** a real CCTV/RTSP camera, the gate-box mode with a real camera, anything on a GPU, night or rain footage,
and the PPE part (paused).

---

## 3. Plan for the GPU machine

### 3.1 Move it cleanly
- [ ] Write a pinned `requirements.txt` and a `Dockerfile` (NVIDIA Container Toolkit), or a service script on Windows.
- [ ] Copy `app/`, `plate_ocr.py`, `models/`, and the scripts you want. PaddleOCR downloads its models into `~/.paddlex` on first
      use, so the machine needs internet once, or copy that folder.
- [ ] **Do not copy** `app/data.db`, `app/webhooks.json` (secrets in plain text) or `app/snaps/`.
- [ ] Check the port: 8010 here; change it if needed.

### 3.2 Switch it to the GPU
- [ ] Install the CUDA builds of PyTorch and PaddlePaddle that match the driver.
- [ ] The code is pinned to the CPU today: every detector call says `device="cpu"`, the OCR is created with
      `enable_mkldnn=False`, and there are thread caps (`CPU_THREADS`, `OMP_NUM_THREADS`). Make the device a setting.
- [ ] Remove the compromises made for the CPU: the lighter reader for the camera, `FULL_MAX_WIDTH = 800`, reading only every
      `CAMERA_RECHECK_S = 1.5` s. With a GPU the accurate reader can run on everything. All of these are named constants
      in `app/engine.py` and `plate_ocr.py`.
- [ ] Export the detector and OCR to TensorRT or ONNX for extra speed, and use half precision.
- [ ] Measure again with `measure_cpu.py`-style checks. No GPU number exists yet; the expectation is roughly 5 to 10 times
      faster, but that is an estimate, not a measurement.

### 3.3 Raise accuracy: train on Indian plates (the biggest lever)
A GPU does not make the same models more accurate; it makes **training** and **bigger models** practical.
- [ ] Run `make_ocr_dataset.py` to cut the plate crops and labels (train/val/test split by file name; the test split is never
      trained on).
- [ ] Fine-tune the **OCR recognition model** on those crops, plus synthetic plates and **footage from your own cameras**.
- [ ] Fine-tune the **plate detector** so two-line bike plates and large plates get a full box. It currently shrinks the box at
      larger input sizes.
- [ ] Rerun `eval_plates.py` and compare with section 2. Set a target (for example 95% exact on clean plates and 85% on
      CCTV-like images) and do not release before it is met.
- [ ] Use the dashboard's "Check this one" reads as a review queue, and feed corrections back as new training data.

### 3.4 Real cameras
- [ ] The video/gate code reads an RTSP stream, tracks vehicles with ByteTrack and uses a box on the frame, but it is untested
      on a real camera. Test it first on a recorded video, then on the live stream.
- [ ] Add reconnection when a stream drops, separate threads for grabbing frames and for reading them (drop old frames), and
      one pipeline per camera.
- [ ] **Camera placement matters as much as the software:** the plate at least about 100 to 150 pixels wide, a fast shutter,
      infrared light at night, a shallow angle, no glare.

### 3.5 Security and reliability
- [ ] **There is no login** on the dashboard, the WebSocket or the webhook settings. Add authentication, put it behind HTTPS
      (a reverse proxy such as nginx or Caddy), and decide who can reach it.
- [ ] Webhooks accept any http/https address. Block internal addresses and consider an allowlist. Store secrets safely, not
      in `webhooks.json`.
- [ ] Limit upload size and file types.
- [ ] Run it as a service that restarts on failure. Add a health check, log rotation, and GPU/CPU monitoring with alerts.
- [ ] Move from SQLite to PostgreSQL if several cameras write at once. Add backups and a disk quota for `snaps/`.

### 3.6 Legal and privacy
- [ ] **Ultralytics YOLO is AGPL-3.0**, and the plate detector weights used here (`morsetechlab/yolov11-license-plate-detection`)
      are AGPL-3.0 too. Commercial use needs a paid licence, or a switch to Apache-licensed models (for example YOLOX or
      PaddleDetection) and weights you trained yourself. Check every model and dataset licence before shipping.
- [ ] The labelled photos come from Zenodo record 13954135/13954136 (CC BY 4.0, attribution required).
- [ ] Plates are personal data under India's DPDP Act: define what is stored, for how long, and who can see it.

### 3.7 Prove it works
- [ ] Add automated tests (the text cleanup, the API, the swap-a-plate case in `test_swap.py`) and run them before each release.
- [ ] Record a day of real footage, check the reads by hand, and measure precision, recall and delay against the target.
- [ ] Keep model versions, so a bad update can be rolled back.

---

## 4. Known limits and decisions to revisit

- **The text cleanup.** It fixes a few common character swaps by position (state letters, district digits, series letters) and
  checks the state code against a list. These are patches; training should make most of them unnecessary.
- **The 800 px limit on large crops** lowers accuracy on huge phone photos (about 28% instead of 49% on the hardest ones) and
  exists only because the slow reader takes 10 to 25 s on a CPU.
- **The "same plate" rule** treats two readings that are 85% similar as the same plate, so a one-character wobble is not a
  new plate. Two real plates that differ by one character in the same spot would be confused.
- **The PPE part** (helmet, vest, goggles, shoes) was paused. The model used is unproven.
- **Gate mode** (only vehicles inside a drawn box) exists in the engine and was tested on one demo video, not on a real gate.

---

## 5. Suggested order

1. Move to the GPU machine and switch the device setting (3.1, 3.2). Rerun `eval_plates.py` for a baseline.
2. Add authentication and HTTPS (3.5), since the app will no longer be only on one PC.
3. Train on Indian plates (3.3) and reach the accuracy target.
4. Test on the real camera (3.4), including night.
5. Settle licences and privacy (3.6) before any commercial use.
6. Automate tests and monitoring (3.5, 3.7), then run a day of real traffic before release.
