"""Gate-box check against a running app: a box over the car should give one record, a box in an empty corner none.

usage: python test_gate.py [port] [video]
"""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8010"
VIDEO = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/Admin/ppe-anpr/samples/anpr_demo.mp4"
BASE = f"http://127.0.0.1:{PORT}"


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def run(name, zone):
    call("/api/reset", {})
    call("/api/zone", zone)
    call("/api/start", {"source": VIDEO, "every": 3, "mode": "gate"})
    t0 = time.time()
    while time.time() - t0 < 400:
        time.sleep(5)
        s = call("/api/status")
        if not s["running"]:
            break
    rows = call("/api/plates?limit=50")
    print(f"== {name}: frames={s['frames']} error={s['error']} records={len(rows)}")
    for r in rows:
        print(f"   id={r['id']} plate={r['plate']!r} reads={r['reads']} status={r['status']} vehicle={'yes' if r['vehicle_snap'] else 'no'}")


run("A: box over the car", {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95})
run("B: small empty corner", {"x1": 0.0, "y1": 0.0, "x2": 0.15, "y2": 0.15})
