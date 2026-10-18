"""Re-score the plate cleanup rules on the saved raw OCR text (no OCR rerun). usage: python tune_cleanup.py"""
import csv
from pathlib import Path

from plate_ocr import clean_plate

rows = list(csv.DictReader(open(Path(__file__).parent / "dataset-indian/eval_results.csv")))
exact = valid = 0
for r in rows:
    pred, ok = clean_plate(r["raw"])
    exact += pred == r["truth"]
    valid += ok
    if pred != r["truth"]:
        print(f"{r['image']:10} truth={r['truth']:11} raw={r['raw']:11} pred={pred}")
print(f"\nexact {exact}/{len(rows)}  valid-format {valid}/{len(rows)}")
