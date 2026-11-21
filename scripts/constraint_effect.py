"""Does the replacement stage respond to the FollowIR instruction change?

For every applied swap, classify the removed document (demoted under the changed
instruction or not) and measure nDCG@10 under the changed qrels before and after
the swap. CPU only.

python scripts/constraint_effect.py --exp-root results/replacement_v2/followir \
    --qrels-root D:\\RETECO\\followir\\domains --out results/replacement_v2/followir/constraint_effect.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

COLS = [("core17", "fo", "exp_followir_dev.json", "candidates_followir_dev.json"),
        ("news21", "fn21", "exp_fn21_dev.json", "candidates_fn21_dev.json"),
        ("robust04", "frob", "exp_frob_dev.json", "candidates_frob_dev.json")]


def read_qrels(path):
    rel = defaultdict(dict)
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 4:
            rel[p[0]][p[2]] = int(p[3])
    return rel


def ndcg(rel, ranked, k=10):
    dcg = 0.0
    for i, d in enumerate(ranked[:k], 1):
        g = rel.get(d, 0)
        if g:
            dcg += (2 ** g - 1) / math.log2(i + 1)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1) if g)
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default="results/replacement_v2/followir")
    ap.add_argument("--qrels-root", default=r"D:\RETECO\followir\domains")
    ap.add_argument("--out", default="results/replacement_v2/followir/constraint_effect.json")
    a = ap.parse_args()
    root = Path(a.exp_root)
    out = {}
    for name, p, exp_name, cand_name in COLS:
        exp = json.loads((root / exp_name).read_text())
        cand = json.loads((root / cand_name).read_text())
        top10 = {(q["domain"], str(q["qid"])): q["frozen_top10"] for q in cand["queries"]}
        chg = read_qrels(Path(a.qrels_root) / ("%s_chg" % p) / "qrels_dev.txt")
        og = read_qrels(Path(a.qrels_root) / ("%s_og" % p) / "qrels_dev.txt")
        per = defaultdict(list)
        swaps = removed_demoted = 0
        base_vals, new_vals = [], []
        for q in exp["pair_details"]:
            key = (q["domain"], str(q["qid"]))
            top = top10.get(key)
            if not top:
                continue
            chosen = [x for x in q["pairs"] if x.get("chosen")]
            ranked = list(top)
            if chosen:
                c = chosen[0]
                slot = int(c["slot_rank"])
                ranked[slot - 1] = c["challenger"]
                swaps += 1
                if og.get(str(q["qid"]), {}).get(c["incumbent"], 0) > 0 and \
                   chg.get(str(q["qid"]), {}).get(c["incumbent"], 0) == 0:
                    removed_demoted += 1
            b = ndcg(chg.get(str(q["qid"]), {}), top)
            n = ndcg(chg.get(str(q["qid"]), {}), ranked)
            base_vals.append(b); new_vals.append(n)
            per[key[0]].append(n - b)
        out[name] = {"queries": len(base_vals), "swaps": swaps,
                     "swaps_removing_demoted_doc": removed_demoted,
                     "ndcg_chg_base": sum(base_vals) / len(base_vals) if base_vals else None,
                     "ndcg_chg_after": sum(new_vals) / len(new_vals) if new_vals else None,
                     "queries_improved": sum(1 for v in per and [] or []),
                     }
        out[name]["ndcg_chg_delta"] = (out[name]["ndcg_chg_after"] - out[name]["ndcg_chg_base"]
                                       if base_vals else None)
        out[name]["queries_improved"] = sum(1 for v in (new_vals[i] - base_vals[i]
                                                        for i in range(len(base_vals))) if v > 1e-12)
        out[name]["queries_harmed"] = sum(1 for v in (new_vals[i] - base_vals[i]
                                                      for i in range(len(base_vals))) if v < -1e-12)
        print(name, {k: v for k, v in out[name].items() if k != "queries"})
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
