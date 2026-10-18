"""Plate text reading (PaddleOCR) and Indian plate cleanup, shared by run.py, the web app and the tests."""
import math
import os
import re
import threading

import cv2

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# SS DD [A-Z]{0,3} NNNN, e.g. MH12AB1234 (series can be absent on older plates); BH series: 22BH1234AA
IN_PLATE = re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{0,3})(\d{4})$|^(\d{2})BH(\d{4})([A-Z]{1,2})$")


def _text_height(poly):
    """True height of a text box: its shorter side. (A tilted plate's box has a misleadingly small y-extent.)"""
    side = lambda a, b: math.hypot(poly[a][0] - poly[b][0], poly[a][1] - poly[b][1])
    return min(side(0, 1), side(1, 2))


# Measured on Indian plates (see bench_ocr_speed.py and bench_ocr_fullsize.py): on small crops, like a plate in a
# webcam frame, the text-finding step of the full pipeline fails often, and reading the crop directly is about 5x
# faster and far more accurate (70% vs 28% exact). On large crops from big photos the full pipeline is better
# (43% vs 26%), so it stays for those.
REC_ONLY_BELOW = 450  # pixels of crop width
FULL_MAX_WIDTH = 800  # "best" mode skips the slow full pipeline on crops wider than this (time limit for the photo tab)
FAST_WIDTH = 400      # "fast" mode shrinks larger crops to this width, the size the recognition-only reader was measured at


class PlateReader:
    def __init__(self, target_width=480, rec_only_below=REC_ONLY_BELOW):
        self.target_width = target_width
        self.rec_only_below = rec_only_below
        from paddleocr import PaddleOCR, TextRecognition
        threads = max(2, (os.cpu_count() or 4) // 2)  # see CPU_THREADS in app/engine.py
        try:
            self._rec = TextRecognition(model_name="PP-OCRv6_medium_rec", enable_mkldnn=False, cpu_threads=threads)
        except TypeError:
            self._rec = TextRecognition(model_name="PP-OCRv6_medium_rec")
        try:
            self._ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                                  use_textline_orientation=False, enable_mkldnn=False, cpu_threads=threads)
        except TypeError:
            self._ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                                  use_textline_orientation=False, enable_mkldnn=False)
        self._lock = threading.Lock()  # the web app can read a photo while the video pipeline is reading plates

    def _read_rec_only(self, crop):
        """Read a small, already-cropped plate with the recognition model alone (no separate text-finding step)."""
        h, w = crop.shape[:2]
        parts = [crop] if w / h > 2.6 else [crop[: h // 2], crop[h // 2:]]  # a squarer plate is two lines: read each
        text, scores = "", []
        with self._lock:
            for part in parts:
                o = self._rec.predict(input=part, batch_size=1)[0]
                text += re.sub(r"[^A-Za-z0-9]", "", o["rec_text"]).upper()
                scores.append(float(o["rec_score"]))
        return text, (sum(scores) / len(scores) if scores else 0.0)

    def read(self, crop, target_width=None, mode="auto"):
        """Return (raw text, mean confidence) for a plate crop. Two-line plates are read top line first.

        mode: "best" reads with both readers and keeps the better result (see pick_better);
              "auto" reads small crops with recognition only and large crops with the full pipeline;
              "fast" always uses recognition only, first shrinking a large crop to FAST_WIDTH (for the live camera,
              where speed matters most); "full" always uses the full pipeline."""
        if crop is None or crop.size == 0:
            return "", 0.0
        if mode == "best":
            # the fast reader first; if it already gives a confident, valid plate that is the answer (measured: 90% vs
            # 92% for always running both, in half the time). Otherwise run the slow reader too and keep the better.
            fast = self.read(crop, mode="fast")
            if clean_plate(fast[0])[1] and fast[1] >= 0.9:
                return fast
            if crop.shape[1] > FULL_MAX_WIDTH:  # the slow reader takes 10-25 s on a crop this large: not worth the wait
                return fast
            return pick_better(self.read(crop, mode="full"), fast)
        if mode == "fast":
            if crop.shape[1] > FAST_WIDTH:
                s = FAST_WIDTH / crop.shape[1]
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            return self._read_rec_only(crop)
        if mode == "auto" and crop.shape[1] < self.rec_only_below:
            return self._read_rec_only(crop)
        # small crops are enlarged so the characters are big enough to read; the cost grows with the square of the
        # enlargement, so it is capped at 3x and aims for target_width pixels rather than a fixed multiple
        tw = self.target_width if target_width is None else target_width
        w = crop.shape[1]
        if w < tw:
            s = min(3.0, tw / w)
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        with self._lock:
            out = self._ocr.predict(crop)
        if not out:
            return "", 0.0
        r = out[0]
        items = [(p, t, s) for p, t, s in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"])]
        if not items:
            return "", 0.0
        # the plate's own characters are its tallest text; the small "IND" / chakra marks on the blue strip of
        # high-security plates are far smaller, so drop anything under half the tallest piece
        tallest = max(_text_height(p) for p, _, _ in items)
        items = [it for it in items if _text_height(it[0]) >= 0.5 * tallest]
        heights = sorted(_text_height(p) for p, _, _ in items)
        line_h = max(heights[len(heights) // 2], 1)
        items.sort(key=lambda it: (round(((it[0][0][1] + it[0][2][1]) / 2) / line_h), it[0][0][0]))
        text = "".join(re.sub(r"[^A-Za-z0-9]", "", t).upper() for _, t, _ in items)
        return text, sum(s for *_, s in items) / len(items)


def pick_better(a, b):
    """Choose between two (raw text, confidence) readings of the same plate, without changing either.

    The two readers fail differently: the full pipeline can drop whole lines of a two-line plate, the recognition-only
    reader can misread a big or busy crop. A reading that forms a valid plate wins; otherwise the longer one wins, because
    dropping characters is the usual failure; otherwise the more confident one."""
    (ta, va), (tb, vb) = clean_plate(a[0]), clean_plate(b[0])
    if va != vb:
        return a if va else b
    if len(ta) != len(tb):
        return a if len(ta) > len(tb) else b
    return a if a[1] >= b[1] else b


# characters OCR commonly swaps, by the kind of position they should be in
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


STATES = {"AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA",
          "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR",
          "TS", "UK", "UP", "WB"}


def _fix_zones(t):
    """Fix confusable characters by zone: state letters, district digits, series letters, last 4 digits."""
    c = list(t)
    series_end = len(c) - 4
    for i in range(len(c)):
        if i < 2:
            c[i] = TO_ALPHA.get(c[i], c[i])
        elif i < 4 or i >= series_end:
            c[i] = TO_DIGIT.get(c[i], c[i])
        elif c[i] == "0":  # a 0 inside the series is usually a D (first letter) or Q (later letters)
            c[i] = "D" if i == 4 else "Q"
        else:
            c[i] = TO_ALPHA.get(c[i], c[i])
    return "".join(c)


def _is_plate(t):
    return bool(IN_PLATE.match(t)) and (t[:2] in STATES or t[2:4] == "BH")


def clean_plate(raw):
    """Normalise raw OCR text to an Indian plate; return (text, is_valid_format).

    The end of the reading is never shortened to make it fit a format: a plate that does not match the usual pattern (a
    bike plate with an extra digit, a custom plate, a misread) is returned whole with is_valid_format False, so the
    person sees everything the reader saw. Junk read IN FRONT of the number (the "IND" mark of the blue strip, a
    sticker, the border) is dropped, up to 3 characters, but only when what follows is then a valid plate."""
    t = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if _is_plate(t):
        return t, True
    for k in (1, 2, 3):
        rest = t[k:]
        fixed = _fix_zones(rest) if 8 <= len(rest) <= 10 else rest
        if _is_plate(fixed):
            return fixed, True
    if 8 <= len(t) <= 10:
        fixed = _fix_zones(t)
        return fixed, _is_plate(fixed)
    return t, False
