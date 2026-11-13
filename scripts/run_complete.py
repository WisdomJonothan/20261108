#!/usr/bin/env python
"""Completion markers for cached evaluation steps.

`ops/final_eval.sh` used to resume on `[ -s "$out" ]`, which happily reuses a
stale run from a different checkpoint, query mode, or query length, and cannot
tell a complete file from one truncated by a killed process.  A step is now
reusable only when

  * the marker `<out>.done.json` exists,
  * the marker's fingerprint equals the current (checkpoint, parameters, inputs),
  * and the output file still has the size and sha1 the marker recorded.

Usage
    run_complete.py stamp --out FILE --ckpt DIR --params "k=v ..." --input FILE...
    run_complete.py check --out FILE --ckpt DIR --params "k=v ..." --input FILE...
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def file_digest(path):
    h = hashlib.sha1()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def dir_digest(path):
    """Model directory as (relpath, size, mtime_ns) per file -- no file reads."""
    h = hashlib.sha1()
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            f = os.path.join(root, name)
            st = os.stat(f)
            h.update(os.path.relpath(f, path).encode("utf-8", "ignore"))
            h.update(b"|%d|%d|" % (st.st_size, st.st_mtime_ns))
    return h.hexdigest()[:16]


def fingerprint(ckpt, params, inputs, stat_inputs=()):
    h = hashlib.sha1()
    h.update(dir_digest(ckpt).encode())
    h.update(b"|")
    h.update(params.encode("utf-8", "ignore"))
    for p in inputs:
        size, sha = file_digest(p)
        h.update(b"|%s|%d|%s" % (os.path.basename(p).encode(), size, sha.encode()))
    for p in stat_inputs:
        st = os.stat(p)
        h.update(b"|stat|%s|%d|%d" % (os.path.basename(p).encode(), st.st_size,
                                      st.st_mtime_ns))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["stamp", "check"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--input", nargs="*", default=[])
    ap.add_argument("--stat-input", nargs="*", default=[],
                    help="large inputs bound by size+mtime instead of a full hash")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    marker = a.out + ".done.json"
    fp = fingerprint(a.ckpt, a.params, a.input, a.stat_input)

    if a.action == "stamp":
        size, sha = file_digest(a.out)
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "ckpt": os.path.abspath(a.ckpt),
                       "params": a.params, "inputs": [os.path.abspath(p) for p in a.input],
                       "size": size, "sha1": sha, "note": a.note,
                       "stamped": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)
        return 0

    if not os.path.isfile(a.out):
        print("no output", file=sys.stderr)
        return 1
    if not os.path.isfile(marker):
        print("no completion marker (old-style output; rerun or adopt explicitly)",
              file=sys.stderr)
        return 1
    try:
        m = json.load(open(marker, encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print("unreadable marker: %s" % exc, file=sys.stderr)
        return 1
    if m.get("fingerprint") != fp:
        print("fingerprint mismatch: marker %s, now %s" % (m.get("fingerprint"), fp),
              file=sys.stderr)
        return 1
    size, sha = file_digest(a.out)
    if size != m.get("size") or sha != m.get("sha1"):
        print("output changed since it was stamped (%d bytes vs %d)"
              % (size, m.get("size")), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
