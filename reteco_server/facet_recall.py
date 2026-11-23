"""Facet recall probe: do the official decomposition steps retrieve gold that the
full query misses?  BM25 (k1=0.9,b=0.4) over each domain corpus, CPU only."""
import argparse, json, os, sys
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
            elif len(parts) == 3 and int(float(parts[2])) > 0:
                rel[parts[0]].add(parts[1])
    return rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    results = {}
    for d in a.domains:
        base = os.path.join(ROOT, "data/track1_tempo", d)
        docs_p = os.path.join(base, "documents.jsonl")
        steps_p = os.path.join(base, f"steps_{a.split}.jsonl")
        qr_q = load_qrels(os.path.join(base, f"qrels_{a.split}.txt"))
        qr_s = load_qrels(os.path.join(base, f"qrels_steps_{a.split}.txt"))
        ids, texts = [], []
        for o in read_jsonl(docs_p):
            ids.append(o.get("id") or o.get("docid"))
            t = o.get("content") or o.get("text") or ""
            ti = o.get("title") or ""
            texts.append((ti + " " + t) if ti else t)
        vec = CountVectorizer(lowercase=True, binary=True, min_df=1, dtype=np.float32,
                              token_pattern=r"(?u)\b\w\w+\b", stop_words=None)
        X = vec.fit_transform(texts)
        vocab = vec.vocabulary_
        N, avgdl = X.shape[0], X.sum(axis=1).mean()
        dl = np.asarray(X.sum(axis=1)).ravel()
        scale = ((a.k1 + 1) / (1 + a.k1 * (1 - a.b + a.b * dl / avgdl))).astype(np.float32)
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)

        def top(term_ids, k):
            if not term_ids:
                return []
            qv = np.zeros(X.shape[1], dtype=np.float32)
            for t in term_ids:
                qv[t] = idf[t]
            s = np.asarray(X @ qv).ravel() * scale
            k = min(k, s.shape[0])
            idx = np.argpartition(-s, k - 1)[:k]
            idx = idx[np.argsort(-s[idx])]
            return [ids[i] for i in idx]

        def terms(txt):
            return [vocab[w] for w in vec.build_analyzer()(txt) if w in vocab]

        n_q = n_q_hit = n_s = n_s_hit = 0
        zero_q = zero_hit = 0
        detail = []
        for o in read_jsonl(steps_p):
            qid = str(o["id"])
            gold = qr_q.get(qid, set())
            n_q += 1
            qhits = top(terms(o["query"]), a.topk)
            qhit = len(gold & set(qhits)) if gold else 0
            n_q_hit += 1 if qhit else 0
            step_hit_any, step_ret = 0, 0
            for st in o.get("steps") or []:
                sid = str(st.get("step_id"))
                sg = qr_s.get(sid, set())
                sh = top(terms(st.get("step_instruction") or st.get("step") or ""), a.topk)
                n_s += 1
                ok = bool(sg & set(sh))
                step_ret += 1 if sg else 0
                if ok:
                    n_s_hit += 1
                step_hit_any = 1 if (ok or step_hit_any) else 0
            if qhit == 0:
                zero_q += 1
                zero_hit += 1 if step_hit_any else 0
            detail.append({"qid": qid, "gold": len(gold), "query_hit": qhit,
                           "steps": len(o.get("steps") or []),
                           "step_hit_any": step_hit_any})
        results[d] = {"queries": n_q, "query_hit": n_q_hit, "steps": n_s,
                      "steps_with_qrels": step_ret, "step_hit": n_s_hit,
                      "query_miss": zero_q, "query_miss_fixed_by_step": zero_hit,
                      "docs": N, "detail": detail}
        print(json.dumps({k: v for k, v in results[d].items() if k != "detail"}),
              flush=True)
    json.dump(results, open(a.out, "w", encoding="utf-8"), indent=1)
    print("FACET_RECALL_DONE", a.out)

if __name__ == "__main__":
    main()
