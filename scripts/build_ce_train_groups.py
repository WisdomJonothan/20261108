#!/usr/bin/env python3
"""Build per-query candidate groups from TRAIN split for cross-encoder fine-tuning.

Reads only train qrels + a train candidate run (dense_<tag>_<domain>_train.trec).
Writes one JSONL per domain: one line per query group.

Group composition (train-time only; the dev split is never read):
  * all positives that appear in the candidate list (label 1)
  * negatives: `--hard-frac` of them taken from the best-ranked non-gold
    candidates (the ones the retriever already confuses), the rest sampled
    uniformly from the remaining candidates
  * groups with zero positives are dropped (they teach nothing under a
    listwise loss) and counted in the meta file

Usage:
  python build_ce_train_groups.py --domains bitcoin economics genealogy history \
      hsm monero politics --k 100 --group-size 20 --hard-frac 0.7 \
      --out-prefix results/ce_train_groups
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_qrels, read_trec_run, load_examples  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_query(text):
    import re
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = t.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
    return re.sub(r"\s+", " ", t).strip()


def build_domain(domain, k, group_size, hard_frac, rng, run_tag, backfill_gold=0):
    data_dir = os.path.join(ROOT, "data", "track1_tempo", domain)
    run_path = os.path.join(ROOT, "runs", f"dense_{run_tag}_{domain}_train.trec")
    qrels = load_qrels(data_dir, "train")
    run = read_trec_run(run_path)

    ids, queries = load_examples(data_dir, "train")
    qtext = dict(zip(ids, queries))

    gold_any = {d for rel in qrels.values() for d, r in rel.items() if r > 0}
    groups, stats = [], {"domain": domain, "gold_any": len(gold_any),
                         "neg_dropped_cross_query_gold": 0,
                         "run": os.path.basename(run_path),
                         "queries": 0, "dropped_no_pos": 0, "positives": 0,
                         "backfilled_queries": 0, "backfilled_pos": 0,
                         "negatives": 0, "hard": 0, "random": 0}
    for qid, ranked in run.items():
        cand = [t[2] for t in ranked][:k]
        rel = qrels.get(qid, {})
        pos = [d for d in cand if rel.get(d, 0) > 0]
        if not pos:
            # ladder step 5: a query whose gold documents the first stage never
            # surfaced is still a valid training signal if we put the labelled
            # positives into the group ourselves (they stay absent at eval time)
            back = [d for d, r in sorted(rel.items()) if r > 0][:backfill_gold]
            back = [d for d in back if d not in cand]
            if backfill_gold and back:
                pos = back
                stats["backfilled_queries"] += 1
                stats["backfilled_pos"] += len(back)
            else:
                stats["dropped_no_pos"] += 1
                continue
        neg_pool = [d for d in cand if rel.get(d, 0) <= 0 and d not in gold_any]
        stats["neg_dropped_cross_query_gold"] += len(
            [d for d in cand if rel.get(d, 0) <= 0 and d in gold_any])
        n_neg = max(0, group_size - len(pos))
        n_hard = int(round(n_neg * hard_frac))
        n_rand = n_neg - n_hard
        # hard negatives: best-ranked non-gold candidates (order = dense rank)
        hard = neg_pool[:n_hard]
        rest = neg_pool[n_hard:]
        rand = rng.sample(rest, min(n_rand, len(rest))) if rest else []
        docs = [{"id": d, "label": 1} for d in pos] + \
               [{"id": d, "label": 0} for d in hard + rand]
        rng.shuffle(docs)
        qfull = qtext.get(qid, "")
        groups.append({
            "qid": qid,
            "domain": domain,
            "query_full": clean_query(qfull),
            "query_title": clean_query(qfull.split("<p>")[0]) or clean_query(qfull),
            "docs": docs,
        })
        stats["queries"] += 1
        stats["positives"] += len(pos)
        stats["negatives"] += len(hard) + len(rand)
        stats["hard"] += len(hard)
        stats["random"] += len(rand)
    return groups, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=20)
    ap.add_argument("--hard-frac", type=float, default=0.7)
    ap.add_argument("--run-tag", default="mxbai",
                    help="candidate run is runs/dense_<tag>_<domain>_train.trec")
    ap.add_argument("--out-prefix", default="results/ce_train_groups")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--backfill-gold", type=int, default=0,
                    help="max labelled positives to inject for queries whose top-k had none")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_stats = []
    for d in args.domains:
        groups, stats = build_domain(d, args.k, args.group_size, args.hard_frac, rng, args.run_tag,
                             args.backfill_gold)
        out = os.path.join(ROOT, f"{args.out_prefix}_{d}.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for g in groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        stats["out"] = os.path.basename(out)
        all_stats.append(stats)
        print(json.dumps(stats, ensure_ascii=False), flush=True)

    total = {k: sum(s[k] for s in all_stats)
             for k in ("queries", "dropped_no_pos", "backfilled_queries", "backfilled_pos",
                       "positives", "negatives", "hard", "random")}
    total["domains"] = len(all_stats)
    total["config"] = {k: getattr(args, k) for k in
                       ("k", "group_size", "hard_frac", "run_tag", "seed", "backfill_gold")}
    meta_path = os.path.join(ROOT, f"{args.out_prefix}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"total": total, "per_domain": all_stats}, f, indent=1)
    print("TOTAL", json.dumps(total, ensure_ascii=False))
    print("meta ->", meta_path)


if __name__ == "__main__":
    main()
