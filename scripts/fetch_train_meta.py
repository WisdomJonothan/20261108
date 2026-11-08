#!/usr/bin/env python3
"""Fetch the small train-split metadata files for one or more domains."""
import os, sys
from huggingface_hub import hf_hub_download

REPO = "DataScience-UIBK/RETECO-SemEval2027"
ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(domains):
    for d in domains:
        tgt = os.path.join(ROOT, "data", "track1_tempo", d)
        os.makedirs(tgt, exist_ok=True)
        for name in ("examples_train.jsonl", "guidance_train.jsonl", "qrels_train.txt",
                     "steps_train.jsonl", "qrels_steps_train.txt"):
            out = os.path.join(tgt, name)
            if os.path.isfile(out) and os.path.getsize(out) > 0:
                continue
            src = f"track1_tempo/{d}/{name}"
            try:
                p = hf_hub_download(REPO, src, repo_type="dataset",
                                    local_dir=os.path.join(ROOT, "tmp", "hfmeta"))
                with open(p, "rb") as f:
                    data = f.read()
                if b"<!DOCTYPE" in data[:200] or len(data) < 20:
                    print("BAD", d, name, len(data))
                    continue
                with open(out, "wb") as f:
                    f.write(data)
                print("ok", d, name, len(data))
            except Exception as e:
                print("FAIL", d, name, e)

if __name__ == "__main__":
    main(sys.argv[1:])
