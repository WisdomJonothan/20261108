"""Parallel ranged download from hf-mirror (8 connections by default)."""
import os
import sys
import threading
import time

import requests

URL = sys.argv[1]
OUT = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8

r = requests.get(URL, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
cr = r.headers.get("Content-Range", "")
total = int(cr.split("/")[-1])
print(f"total={total} bytes", flush=True)

parts = []
step = (total + N - 1) // N


def fetch(i: int, start: int, end: int) -> None:
    path = f"{OUT}.part{i}"
    for attempt in range(6):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        s = start + have
        if s > end:
            return
        try:
            with requests.get(
                URL, headers={"Range": f"bytes={s}-{end}"}, stream=True, timeout=120
            ) as resp:
                resp.raise_for_status()
                with open(path, "ab") as f:
                    for chunk in resp.iter_content(1 << 16):
                        f.write(chunk)
            if os.path.getsize(path) == end - start + 1:
                return
        except Exception as exc:  # noqa: BLE001
            print(f"part{i} attempt{attempt}: {exc}", flush=True)
            time.sleep(2)


threads = []
for i in range(N):
    start = i * step
    end = min(total - 1, start + step - 1)
    t = threading.Thread(target=fetch, args=(i, start, end))
    t.start()
    threads.append((i, t))

for i, t in threads:
    t.join()
    print(f"part{i} done {os.path.getsize(f'{OUT}.part{i}')}", flush=True)

with open(OUT, "wb") as out:
    for i in range(N):
        with open(f"{OUT}.part{i}", "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                out.write(b)
        os.remove(f"{OUT}.part{i}")

print("final size", os.path.getsize(OUT), "expected", total)
