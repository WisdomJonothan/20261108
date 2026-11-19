"""Semantic one-swap pilot on official-train held-out queries."""
import argparse
import json
import re
from html import unescape
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from coverage_cf_bound import ndcg10, read_qrels, read_steps
from eval_replacement_pilot import terms
from lib_reteco import read_trec_run


THRESHOLDS = (None, 0., .025, .05, .1, .2, .4)


def embed(texts, tokenizer, model, device):
    output = []
    for start in range(0, len(texts), 32):
        tokens = tokenizer(texts[start:start+32], padding=True, truncation=True,
                           max_length=160, return_tensors='pt').to(device)
        hidden = model(**tokens).last_hidden_state.float()
        mask = tokens.attention_mask.unsqueeze(-1)
        avg = (hidden * mask).sum(1) / mask.sum(1)
        output.append(torch.nn.functional.normalize(avg, dim=1).cpu().numpy())
    return np.concatenate(output)


def passages(body, query_terms):
    words = strip_html(body).split()
    windows = [' '.join(words[i:i+112]) for i in range(0, len(words), 80)] or ['']
    best = sorted(windows, key=lambda w: -len(terms(w) & query_terms))[:2]
    return best


def strip_html(text):
    return unescape(re.sub(r'<[^>]+>', ' ', text))


def propose(doc_ids, similarities):
    full = similarities[:, 0]
    full = (full - full.mean()) / max(float(full.std()), 1e-8)
    evidence = similarities[:, 1:]
    evidence = (evidence - evidence.mean(axis=0)) / np.maximum(evidence.std(axis=0), 1e-8)
    direct = []
    contextual = []
    for slot in range(7, 10):
        retained = np.max(np.delete(evidence[:10], slot, axis=0), axis=0)
        before = np.maximum(retained, evidence[slot])
        for incoming in range(10, len(doc_ids)):
            margin = (full[incoming] - full[slot]) / np.log2(slot + 2)
            change = np.mean(np.maximum(retained, evidence[incoming]) - before)
            direct.append((float(margin), slot, incoming))
            contextual.append((float(margin + .5 * change), slot, incoming))
    return {'direct': max(direct, key=lambda x: x[0]),
            'contextual': max(contextual, key=lambda x: x[0])}


def measure(rank, gold, steps):
    return {'ndcg': ndcg10(rank, gold),
            'coverage': (sum(bool(set(rank) & docs) for docs in steps.values()) / len(steps))
            if steps else None}


def crossfit(records):
    groups = {domain: [row for row in records if row['domain'] == domain]
              for domain in sorted({row['domain'] for row in records})}
    rng = np.random.default_rng(20261119)
    assignments = {d: rng.permutation(len(rows)) % 2 for d, rows in groups.items()}
    predictions = {'direct': {}, 'contextual': {}}
    selected = []
    for fold in range(2):
        fit = [[row for idx, row in enumerate(rows) if len(rows) == 1 or assignments[d][idx] != fold]
               for d, rows in groups.items()]
        choices = {}
        for method, results in predictions.items():
            fit_scores = [np.mean([np.mean([row[method][i]['ndcg'] for row in subset])
                                   for subset in fit]) for i in range(len(THRESHOLDS))]
            best = max(range(len(THRESHOLDS)), key=lambda i: (
                fit_scores[i], float('inf') if i == 0 else THRESHOLDS[i]))
            choices[method] = THRESHOLDS[best]
            for d, rows in groups.items():
                if len(rows) > 1:
                    for idx, row in enumerate(rows):
                        if assignments[d][idx] == fold:
                            results[d, row['qid']] = row[method][best]
        selected.append(choices)
    report = {}
    for method in ('baseline', *predictions):
        scores, coverage = [], []
        swaps = {'helpful': 0, 'harmful': 0, 'neutral': 0}
        for d, rows in groups.items():
            if len(rows) == 1:
                continue
            outcomes = [row['baseline'] if method == 'baseline'
                        else predictions[method][d, row['qid']] for row in rows]
            scores.append(np.mean([x['ndcg'] for x in outcomes]))
            values = [x['coverage'] for x in outcomes if x['coverage'] is not None]
            if values:
                coverage.append(np.mean(values))
            for row, outcome in zip(rows, outcomes):
                if outcome.get('swap'):
                    delta = outcome['ndcg'] - row['baseline']['ndcg']
                    swaps['helpful' if delta > 1e-12 else 'harmful' if delta < -1e-12 else 'neutral'] += 1
        report[method] = {'macro_ndcg': float(np.mean(scores)),
                          'macro_step_coverage': float(np.mean(coverage)) if coverage else None,
                          'swaps': swaps}
    return {'fold_thresholds': selected, 'metrics': report,
            'fold_ids': {d: {row['qid']: int(assignments[d][i]) for i, row in enumerate(rows)}
                         for d, rows in groups.items()},
            'excluded_singleton_domains': [d for d, rows in groups.items() if len(rows) == 1]}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.root
    model_path = root / 'models/e5-base-v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    if device == 'cuda':
        model.half()
    split = json.loads((root / 'results/fusion100_split.json').read_text(encoding='utf-8'))['eval']
    records = []
    for domain, holdout in split.items():
        data = root / 'data/track1_tempo' / domain
        qids = set(holdout[:2] if args.smoke else holdout)
        queries = {row['id']: row['query'] for row in map(json.loads, (data / 'examples_train.jsonl')
                   .read_text(encoding='utf-8').splitlines()) if row['id'] in qids}
        run = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_train_head20.trec')
        pools = {qid: [doc for _, _, doc in run[qid][:20]] for qid in sorted(qids)}
        assert set(queries) == qids and all(len(set(docs)) == 20 for docs in pools.values())
        needed = {doc for docs in pools.values() for doc in docs}
        corpus = {}
        with (data / 'documents.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['id'] in needed:
                    corpus[row['id']] = row['content']
                    if len(corpus) == len(needed):
                        break
        assert set(corpus) == needed
        gold = read_qrels(data / 'qrels_train.txt')
        steps = read_steps(data / 'qrels_steps_train.txt')
        for qid, docs in pools.items():
            query = strip_html(re.sub(r'</?(?:p|blockquote|br)\b[^>]*>', '\n',
                                      queries[qid], flags=re.IGNORECASE))
            clauses = [s.strip() for s in re.split(r'[.!?\n]+', query) if len(terms(s)) >= 3][:8] or [query]
            query_embeddings = embed(['query: ' + s for s in [query] + clauses], tokenizer, model, device)
            texts = [passages(corpus[doc], terms(query)) for doc in docs]
            doc_embeddings = embed(['passage: ' + t for pair in texts for t in pair], tokenizer, model, device)
            sims = doc_embeddings @ query_embeddings.T
            best_sims = np.stack([x.max(0) for x in np.split(sims, np.cumsum(list(map(len, texts)))[:-1])])
            assert np.isfinite(best_sims).all()
            proposals = propose(docs, best_sims)
            baseline = measure(docs[:10], gold[qid], steps[qid])
            result = {'domain': domain, 'qid': qid, 'baseline': baseline, 'comparisons_per_method': 30}
            for method, (score, slot, incoming) in proposals.items():
                rank = docs[:10].copy()
                rank[slot] = docs[incoming]
                swapped = {**measure(rank, gold[qid], steps[qid]), 'swap': {
                    'rank': slot + 1, 'incoming': docs[incoming], 'removed': docs[slot]}}
                result[method] = [swapped if threshold is not None and score > threshold
                                  else baseline for threshold in THRESHOLDS]
                result[method + '_utility'] = score
            records.append(result)
        print(domain, len(queries), device, flush=True)
        if args.smoke:
            return
    assert len(records) == 214
    result = {'protocol': 'Official train heldout, dense top20. Two shared passages per document at most. Both policies use the same E5 embeddings and 30 candidate comparisons. Only the contextual policy adds a fixed marginal clause-coverage term. Abstention thresholds selected on opposite fold by nDCG, then measured out of fold. Singleton domain fits only.',
              'threshold_grid': THRESHOLDS, 'queries': records, 'crossfit': crossfit(records)}
    (root / 'results/semantic_replacement_pilot.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['crossfit']['metrics']), flush=True)


if __name__ == '__main__':
    main()
