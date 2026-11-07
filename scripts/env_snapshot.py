#!/usr/bin/env python3
"""Record the exact program versions / hardware / data fingerprints of Phase 0."""
import hashlib
import json
import os
ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platform
import subprocess
import sys
from importlib import metadata

PKGS = ["pyserini", "pyjnius", "gensim", "numpy", "scipy", "pandas", "torch",
        "transformers", "huggingface-hub", "sentence-transformers", "tqdm"]
FILES = [
    os.path.join(ROOT, "data", "track1_tempo", "iota", "documents.jsonl",
    os.path.join(ROOT, "data", "track1_tempo", "bitcoin", "documents.jsonl",
    os.path.join(ROOT, "data", "track1_tempo", "iota", "qrels_dev.txt",
    os.path.join(ROOT, "data", "track1_tempo", "bitcoin", "qrels_dev.txt",
    os.path.join(ROOT, "data", "track1_tempo", "iota", "examples_dev.jsonl",
    os.path.join(ROOT, "data", "track1_tempo", "bitcoin", "examples_dev.jsonl",
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    out = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {},
        "java": None,
        "cuda": None,
        "data_sha256": {},
    }
    for p in PKGS:
        try:
            out["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out["packages"][p] = None
    try:
        out["java"] = subprocess.run(
            ["java", "-version"], capture_output=True, text=True).stderr.strip()
    except Exception as e:  # noqa: BLE001
        out["java"] = f"error: {e}"
    try:
        import torch
        out["cuda"] = {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cuda": torch.version.cuda,
        }
    except Exception as e:  # noqa: BLE001
        out["cuda"] = f"error: {e}"
    for f in FILES:
        if os.path.isfile(f):
            out["data_sha256"][f] = sha256(f)
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "env.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "data_sha256"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
