# ANPR: Indian number plate reader

Reads Indian number plates from **photos** and from a **live camera**, records each plate, and pushes every recorded
plate to a **WebSocket** and to **webhooks**. It also contains an older video/gate-box mode and a PPE (helmet, vest,
goggles, shoes) mode that the current dashboard does not use.

> **Status:** a working prototype, not production-grade. It reads clear plates well and large phone photos poorly.
> See [PRODUCTION_PLAN.md](PRODUCTION_PLAN.md) for the measured accuracy, the known limits and what to do next.

## Run it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.server:app --host 127.0.0.1 --port 8010
```

Then open http://127.0.0.1:8010. On Windows, `start_app.bat` does the last step and opens the browser.

- **Photo tab:** drop one or more photos.
- **Camera tab:** choose a camera (laptop or phone) and press Start camera. Browsers allow the camera on `localhost` only
  (or over HTTPS).
- **Live events:** the WebSocket address and webhooks are on the same page.

The first start downloads the OCR models (PaddleOCR) into `~/.paddlex`, so it needs internet once.

## What is in this repository

| Path | What it is |
|---|---|
| `app/` | The web server (`server.py`), the detection engine (`engine.py`) and the dashboard (`static/`) |
| `plate_ocr.py` | The OCR reader and the plate text cleanup |
| `models/` | Plate detector weights (YOLO) and a PPE model |
| `dataset-indian/` | 1,700 labelled Indian plate photos and `number_plate.xlsx` with the correct plates |
| `ocr-dataset/` | Plate crops cut from those photos for training |
| `samples/`, `dashboard-sample/` | Small test inputs |
| `repos/` | Two third-party projects cloned for reference (see their own licences) |
| `eval_plates.py` and `bench_*.py` | Accuracy and speed tests |
| `PRODUCTION_PLAN.md` | Interfaces, measurements, and the plan for a GPU machine |

## Licences and sources: read before reuse

- **Ultralytics YOLO is AGPL-3.0**, and the plate detector weights in `models/plate/`
  (from `morsetechlab/yolov11-license-plate-detection` on Hugging Face) are AGPL-3.0 too. Commercial use needs a licence
  or different models.
- **Photos in `dataset-indian/`** come from the Zenodo dataset "Number Plate Number Identification Dataset"
  (record 13954135), licensed **CC BY 4.0**: attribution is required.
- **`repos/`** holds third-party code and weights with their own licences.
- `samples/anpr_demo.mp4` is a demo video from a third-party project; its licence is unknown.
- This repository contains photos of real vehicles and their number plates. Number plates are personal data
  (India's DPDP Act); think before keeping a public copy.
- No licence has been chosen for the code written here.
