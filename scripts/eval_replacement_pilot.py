"""Fixed-policy one-swap pilot using frozen CE and text-based evidence coverage."""
import argparse
import json
import math
import re
from collections import Counter
from html import unescape
from pathlib import Path

import numpy as np

from coverage_cf_bound import DOMAINS, ndcg10, read_qrels, read_steps
from lib_reteco import read_trec_run


WORDS = re.compile(r"[a-z0-9]+")
STOP = set("the and that this with have what which when where does from were will "
           "would could should about there their into some more been being they "
           "then than them can how are was has did you your these those".split())


def terms(text):
    return {w for w in WORDS.findall(text.lower()) if w not in STOP and len(w) >= 3}


def evidence(query, docs):
    clean = unescape(re.sub(r"<[^>]+>", "\n", query))
    clauses = [terms(piece) for piece in re.split(r"[.!?\n]+", clean)]
    clauses = list({frozenset(clause): None for clause in clauses if len(clause) >= 3})
    if not clauses:
        clauses = [terms(clean)]
    query_terms = set().union(*clauses)
    windows = {}
    frequency = Counter()
    for doc, body in docs.items():
        words = WORDS.findall(body.lower())
        windows[doc] = [set(words[i:i + 96]) & query_terms for i in range(0, len(words), 64)] or [set()]
        frequency.update(set().union(*windows[doc]))
    weights = {w: 1 + math.log((len(docs) + 1) / (frequency[w] + 1)) for w in query_terms}
    # Best short passage match per query clause. This is a lexical proxy.
    return {doc: np.array([
        max(sum(weights[w] for w in (clause & window)) for window in passages)
        / (sum(weights[w] for w in clause) or 1)
        for clause in clauses
    ]) for doc, passages in windows.items()}


def choose_swap(top, incoming, scores, support):
    if not incoming:
        return None, None
    direct, contextual = (0., None), (0., None)
    for slot in range(7, 10):
        removed = top[slot]
        remaining = np.max([support[doc] for doc in top if doc != removed], axis=0)
        before = np.maximum(remaining, support[removed])
        discount = 1 / math.log2(slot + 2)
        for doc in incoming:
            margin = (scores[doc] - scores[removed]) * discount
            coverage_delta = float(np.mean(np.maximum(remaining, support[doc]) - before))
            if margin > direct[0]:
                direct = (margin, (slot, doc))
            value = margin + 0.5 * coverage_delta
            if value > contextual[0]:
                contextual = (value, (slot, doc))
    return direct[1], contextual[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    rows = []
    for domain in DOMAINS:
        data = root / 'data/track1_tempo' / domain
        baseline = read_trec_run(root / 'runs' / f'sub_sft_e2bc_{domain}_dev.trec')
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        ce = read_trec_run(root / 'results/server_review_20261116/runs'
                           / f'fusion100_full_ce_s17_20261115_{domain}_dev_ce.trec')
        queries = {}
        with (data / 'examples_dev.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                queries[item['id']] = item['query']
        assert set(queries) == set(baseline) == set(ce) == set(dense)
        pools = {}
        for q in queries:
            top = [doc for _, _, doc in baseline[q][:10]]
            union = dict.fromkeys(doc for run in (dense, ce) for _, _, doc in run[q][:20])
            pools[q] = (top, [doc for doc in union if doc not in top])
        needed = {doc for top, incoming in pools.values() for doc in top + incoming}
        corpus = {}
        with (data / 'documents.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if item['id'] in needed:
                    corpus[item['id']] = item['content']
                    if len(corpus) == len(needed):
                        break
        assert set(corpus) == needed
        gold = read_qrels(data / 'qrels_dev.txt')
        steps = read_steps(data / 'qrels_steps_dev.txt')
        for q, query in queries.items():
            top, incoming = pools[q]
            raw = {doc: score for _, score, doc in ce[q]}
            values = np.array(list(raw.values()))
            scores = {doc: (raw[doc] - values.mean()) / max(values.std(), 1e-8)
                      for doc in top + incoming}
            support = evidence(query, {doc: corpus[doc] for doc in top + incoming})
            direct, contextual = choose_swap(top, incoming, scores, support)
            baseline_ndcg = ndcg10(top, gold[q])
            endpoints = {}
            for name, swap in [('baseline', None), ('direct', direct), ('contextual', contextual)]:
                ranking = top.copy()
                if swap:
                    ranking[swap[0]] = swap[1]
                endpoints[name] = {
                    'swap': {'rank': swap[0] + 1, 'incoming': swap[1], 'removed': top[swap[0]]} if swap else None,
                    'ndcg': ndcg10(ranking, gold[q]),
                    'step_coverage': (sum(bool(set(ranking) & docs) for docs in steps[q].values())
                                      / len(steps[q])) if steps[q] else None,
                }
            eligible = any(doc in gold[q] for doc in incoming)
            best = max([baseline_ndcg] + [
                ndcg10(top[:i] + [doc] + top[i + 1:], gold[q])
                for i in range(7, 10) for doc in incoming if doc in gold[q]
            ]) if eligible else baseline_ndcg
            rows.append({'domain': domain, 'qid': q, 'comparisons_per_policy': 3 * len(incoming),
                         'oracle': best, 'policies': endpoints})
        print(domain, len(queries), flush=True)
    assert len(rows) == 519
    summary = {}
    for policy in ('baseline', 'direct', 'contextual'):
        per_domain = {}
        counts = Counter()
        for domain in DOMAINS:
            subset = [r for r in rows if r['domain'] == domain]
            scores = [r['policies'][policy]['ndcg'] for r in subset]
            coverage = [r['policies'][policy]['step_coverage'] for r in subset
                        if r['policies'][policy]['step_coverage'] is not None]
            per_domain[domain] = {'ndcg': float(np.mean(scores)),
                                  'step_coverage': float(np.mean(coverage)) if coverage else None}
            for r in subset:
                choice = r['policies'][policy]
                if choice['swap']:
                    delta = choice['ndcg'] - r['policies']['baseline']['ndcg']
                    counts['helpful' if delta > 1e-12 else 'harmful' if delta < -1e-12 else 'neutral'] += 1
        summary[policy] = {
            'macro_ndcg': float(np.mean([v['ndcg'] for v in per_domain.values()])),
            'macro_step_coverage': float(np.mean([v['step_coverage'] for v in per_domain.values()
                                                  if v['step_coverage'] is not None])),
            'swap_counts': dict(counts), 'per_domain': per_domain}
    assert abs(summary['baseline']['macro_ndcg'] - .35809852326897723) < 1e-9
    output = {
        'protocol': 'Exploratory dev pilot. Both policies consider identical replacements at ranks 8–10, using the union of source top20 outside the frozen top10. Direct uses frozen CE margin discounted by rank; contextual adds 0.5 times the change in lexical clause coverage. Choices abstain at nonpositive net utility. Qrels and step annotations are evaluation only. No tuning.',
        'summary': summary, 'queries': rows,
        'oracle_macro': float(np.mean([np.mean([r['oracle'] for r in rows if r['domain'] == d])
                                       for d in DOMAINS])),
        'queries_with_positive_slot_oracle': sum(
            r['oracle'] > r['policies']['baseline']['ndcg'] + 1e-12 for r in rows),
        'total_pair_comparisons_per_policy': sum(r['comparisons_per_policy'] for r in rows),
    }
    rng = np.random.default_rng(20261119)
    comparisons = {}
    for reference in ('direct', 'baseline'):
        deltas = np.array([summary['contextual']['per_domain'][domain]['ndcg']
                           - summary[reference]['per_domain'][domain]['ndcg']
                           for domain in sorted(DOMAINS)])
        draws = deltas[rng.integers(len(deltas), size=(10000, len(deltas)))].mean(axis=1)
        comparisons['contextual_' + reference] = {
            'delta': float(deltas.mean()),
            'domain_bootstrap_95': np.quantile(draws, [.025, .975]).tolist(),
        }
    output['exploratory_paired_comparisons'] = comparisons
    (root / 'results/replacement_pilot.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: {m: v for m, v in s.items() if m != 'per_domain'}
                      for k, s in summary.items()}))
    print('Matched-slot oracle:', output['oracle_macro'])


# --------------------------------------------------------------------------
# Oracle over the frozen candidate/comparison list (step-2 ceiling).
# Reads the SAME file the replacement policies read.
# --------------------------------------------------------------------------
def _read_json(path):
    import json as _json
    with open(path, encoding="utf-8") as fh:
        return _json.load(fh)


def _swap_ndcg(top, slot, incoming, gold):
    ranking = list(top)
    ranking[slot] = incoming
    return ndcg10(ranking, gold)


def oracle_fixed(argv=None):
    import json as _json
    import argparse as _argparse
    import numpy as _np
    from ir_metrics import ndcg_at_k as _official

    ap = _argparse.ArgumentParser(description="Oracle over the frozen candidate list")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/replacement_v2/oracle_fixed_dev.json")
    args = ap.parse_args(argv)
    root = args.root
    payload = _read_json(args.candidates)
    rows = payload["queries"]
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)

    per_query = []
    for domain, subset in by_domain.items():
        gold = read_qrels(root / "data/track1_tempo" / domain / f"qrels_{args.split}.txt")
        for row in subset:
            q, top = row["qid"], row["frozen_top10"]
            g = gold[q]
            base_lex = ndcg10(top, g)
            base_off = _official(top, g, 10)
            assert abs(base_lex - base_off) < 1e-12, (q, base_lex, base_off)
            best = {"delta": 0.0, "pair": None}
            n_pos = n_neg = 0
            for pair in row["pairs"]:
                val = _swap_ndcg(top, pair["slot_rank"] - 1, pair["incoming_docid"], g)
                delta = val - base_off
                if delta > 1e-12:
                    n_pos += 1
                elif delta < -1e-12:
                    n_neg += 1
                if delta > best["delta"] + 1e-12:
                    best = {"delta": delta, "pair": pair, "ndcg": val}
            per_query.append({
                "qid": q, "domain": domain, "baseline": base_off,
                "n_pairs": len(row["pairs"]), "n_pairs_positive": n_pos,
                "n_pairs_negative": n_neg,
                "oracle_ndcg": base_off + best["delta"],
                "oracle_delta": best["delta"],
                "oracle_swap": (None if best["pair"] is None else {
                    "slot_rank": best["pair"]["slot_rank"],
                    "removed_docid": best["pair"]["removed_docid"],
                    "incoming_docid": best["pair"]["incoming_docid"],
                    "incoming_best_rank": best["pair"]["incoming_best_rank"],
                    "incoming_best_source": best["pair"]["incoming_best_source"],
                }),
            })

    domains = sorted(by_domain)
    per_domain = {}
    for d in domains:
        sub = [r for r in per_query if r["domain"] == d]
        per_domain[d] = {
            "n": len(sub),
            "baseline": float(_np.mean([r["baseline"] for r in sub])),
            "oracle": float(_np.mean([r["oracle_ndcg"] for r in sub])),
            "gain": float(_np.mean([r["oracle_delta"] for r in sub])),
            "queries_with_beneficial_pair": int(sum(r["n_pairs_positive"] > 0 for r in sub)),
        }
    macro_base = float(_np.mean([v["baseline"] for v in per_domain.values()]))
    macro_oracle = float(_np.mean([v["oracle"] for v in per_domain.values()]))
    out = {
        "candidate_file": str(args.candidates),
        "protocol": payload["protocol"],
        "macro_baseline": macro_base,
        "macro_oracle": macro_oracle,
        "macro_gain": macro_oracle - macro_base,
        "n_queries": len(per_query),
        "n_pairs": sum(r["n_pairs"] for r in per_query),
        "queries_with_beneficial_pair": int(sum(r["n_pairs_positive"] > 0 for r in per_query)),
        "pairs_positive": int(sum(r["n_pairs_positive"] for r in per_query)),
        "pairs_negative": int(sum(r["n_pairs_negative"] for r in per_query)),
        "sum_best_single_delta": float(sum(r["oracle_delta"] for r in per_query)),
        "per_domain": per_domain,
        "per_query": per_query,
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(out, indent=1), encoding="utf-8")
    print(_json.dumps({k: v for k, v in out.items() if k not in ("per_query", "per_domain")},
                      indent=1))
    for d in domains:
        v = per_domain[d]
        print(f"{d:10s} n={v['n']:3d} base={v['baseline']:.4f} oracle={v['oracle']:.4f} "
              f"gain={v['gain']:+.4f} q_with_gain={v['queries_with_beneficial_pair']}")
    print("wrote", out_path)


if __name__ == '__main__':
    import sys as _sys
    if '--mode' in _sys.argv and _sys.argv[_sys.argv.index('--mode') + 1] == 'oracle-fixed':
        _sys.argv = [a for i, a in enumerate(_sys.argv)
                     if a != '--mode' and not (i and _sys.argv[i - 1] == '--mode')]
        oracle_fixed()
    else:
        main()
