#!/usr/bin/env python3
"""Fail fast when a CE training run used the wrong base model.

The 2026-11-09 OOF rerun silently trained ``models/bge-reranker-base`` (278 M)
because the driver omitted ``--model``; 25 GPU-minutes plus a dev rerank were
spent on numbers that measured a weaker model.  Directory names are not
evidence — the ``data_loaded`` event in ``results/ce_ft/<tag>/train_log.jsonl``
is.

    python check_ce_model.py --tag v2m3_pool8_e1b --expect v2-m3 --params 567.8

Exit code 0 only when every check passes.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expect", default="v2-m3", help="substring of the model path")
    ap.add_argument("--params", type=float, default=None,
                    help="expected parameter count in millions (+/- 2)")
    args = ap.parse_args()

    log = os.path.join(ROOT, "results", "ce_ft", args.tag, "train_log.jsonl")
    if not os.path.isfile(log):
        print(f"FAIL no log: {log}")
        return 2
    rec = None
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "data_loaded":
                rec = r
                break
    if rec is None:
        print(f"FAIL no data_loaded event in {log}")
        return 2

    model = str(rec.get("model", ""))
    params = rec.get("params_m")
    ok = args.expect in model
    print(f"model   = {model}")
    print(f"params  = {params} M")
    if args.params is not None and isinstance(params, (int, float)):
        ok = ok and abs(params - args.params) <= 2.0
    if not ok:
        print(f"FAIL expected a model containing {args.expect!r}"
              + (f" with ~{args.params} M parameters" if args.params else ""))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
