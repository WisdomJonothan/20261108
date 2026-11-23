"""Build a facet (step-level BM25) candidate run per domain and report pool coverage
against the official dev qrels, including the zero-score queries."""
import argparse, json, os
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ROOT = os.path.expanduser("~/RETECO")

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_qrels(p):
    rel = defaultdict(set)
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and int(float(parts[3])) > 0:
                rel[parts[0]].add(parts[2])
    return rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--out-tag", default="facetstep")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    summary = {}
    for d in a.domains:
        base = os.path.join(ROOT, "data/track1_tempo", d)
        ids, texts = [], []
        for o in read_jsonl(os.path.join(base, "documents.jsonl")):
            ids.append(o.get("id") or o.get("docid"))
            ti = o.get("title") or ""
            t = o.get("content") or o.get("text") or ""
            texts.append((ti + " " + t) if ti else t)
        vec = CountVectorizer(lowercase=True, binary=True, dtype=np.float32,
                              token_pattern=r"(?u)\b\w\w+\b")
        X = vec.fit_transform(texts)
        vocab = vec.vocabulary_
        N, avgdl = X.shape[0], X.sum(axis=1).mean()
        dl = np.asarray(X.sum(axis=1)).ravel()
        scale = ((a.k1 + 1) / (1 + a.k1 * (1 - a.b + a.b * dl / avgdl))).astype(np.float32)
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        an = vec.build_analyzer()

        def terms(txt):
            return [vocab[w] for w in an(txt) if w in vocab]

        def scores(txt):
            tt = terms(txt)
            if not tt:
                return None
            qv = np.zeros(X.shape[1], dtype=np.float32)
            for t in tt:
                qv[t] = idf[t]
            return np.asarray(X @ qv).ravel() * scale

        gold = load_qrels(os.path.join(base, f"qrels_{a.split}.txt"))
        rows, nq, cover, cover_any, nsteps = [], 0, 0, 0, 0
        for o in read_jsonl(os.path.join(base, f"steps_{a.split}.jsonl")):
            qid = str(o["id"]); nq += 1
            best = None
            for st in (o.get("steps") or []):
                s = scores(st.get("step_instruction") or st.get("step") or "")
                if s is None:
                    continue
                nsteps += 1
                best = s if best is None else np.maximum(best, s)
            if best is None:
                best = scores(o["query"])
                if best is None:
                    continue
            k = min(a.topk, best.shape[0])
            idx = np.argpartition(-best, k - 1)[:k]
            idx = idx[np.argsort(-best[idx])]
            g = gold.get(qid, set())
            hit = len(g & {ids[i] for i in idx})
            cover += hit
            cover_any += 1 if hit else 0
            for r, i in enumerate(idx, 1):
                rows.append((qid, ids[i], 1.0 / r, r))
        outp = os.path.join(ROOT, "runs", f"{a.out_tag}_{d}_{a.split}.trec")
        with open(outp, "w", encoding="utf-8") as f:
            for qid, docid, sc, r in rows:
                f.write(f"{qid} Q0 {docid} {r} {sc:.6f} {a.out_tag}\n")
        summary[d] = {"docs": N, "queries": nq, "steps": nsteps,
                      "pool_gold": cover, "pool_queries_with_gold": cover_any,
                      "run": outp}
        print(json.dumps({k: v for k, v in summary[d].items() if k != "run"}), flush=True)
    json.dump(summary, open(os.path.join(ROOT, "results/analysis/facet_pool_dev.json"), "w"), indent=1)
    print("FACET_POOL_DONE")

if __name__ == "__main__":
    main()
