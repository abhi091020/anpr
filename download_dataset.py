"""Download every image from the Zenodo Indian number plate dataset (CC BY 4.0) into dataset-indian/.

usage: python download_dataset.py [workers]
Polite by design: few connections, exponential backoff when Zenodo answers 429, and each file is written
to a temp name then renamed, so an interruption never leaves a half-written image. Safe to rerun.
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

URL = "https://zenodo.org/api/records/13954136/files/number_plate.zip/content"
OUT = Path(__file__).parent / "dataset-indian"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

with RemoteZip(URL) as z:
    names = [n for n in z.namelist() if n.startswith("images/") and not n.endswith("/")]
todo = [n for n in names if not (OUT / n).exists()]
print(f"{len(names)} images, {len(todo)} to download, {WORKERS} workers", flush=True)

local = threading.local()
done = 0
lock = threading.Lock()


def fetch(n):
    global done
    dest = OUT / n
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        try:
            if not hasattr(local, "z"):
                local.z = RemoteZip(URL)
            data = local.z.read(n)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            break
        except Exception as e:
            if hasattr(local, "z"):
                del local.z
            wait = min(10 * 2 ** attempt, 300)
            print(f"retry {n} in {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
    else:
        print(f"GAVE UP on {n}", flush=True)
        return
    with lock:
        done += 1
        if done % 50 == 0:
            print(f"{done}/{len(todo)}", flush=True)


with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(fetch, todo))
print("done", flush=True)
