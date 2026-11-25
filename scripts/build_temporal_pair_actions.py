"""Build head and admission proposals from existing RETECO rankings and scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import z
from replay_anchor_fallback import index
from replay_evidence_query_router import cached_queries


def actions(column, ranked, source):
    docs = list(column)
    evidence = {mode: dict(zip(docs, z([scores[d] for d in docs]), strict=True))
                for mode, scores in source.items()}
    head = ranked[1:6]
    leads = list(dict.fromkeys(max(head, key=vote.__getitem__) for vote in evidence.values()))
    pairs = []
    for doc in leads:
        new = ranked.copy()
        previous = ranked.index(doc)
        new[0], new[previous] = new[previous], new[0]
        pairs.append({'type': 'head', 'incumbent': ranked[0], 'challenger': doc, 'slot': 0,
                      'from_slot': previous, 'ranking': new,
                      'evidence_margin': [v[doc]-v[ranked[0]] for v in evidence.values()]})
    combined = {d: np.mean([v[d] for v in evidence.values()]) for d in docs}
    candidate = max((d for d in docs if d not in ranked), key=combined.__getitem__)
    position = min(range(7, 10), key=lambda p: combined[ranked[p]])
    new = ranked.copy()
    new[position] = candidate
    pairs.append({'type': 'admission', 'incumbent': ranked[position], 'challenger': candidate,
                  'slot': position, 'from_slot': 10, 'ranking': new,
                  'evidence_margin': [v[candidate]-v[ranked[position]] for v in evidence.values()]})
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'repair', 'data-root', 'output', 'audit'):
        ap.add_argument(f'--{name}', type=Path, required=True)
    ap.add_argument('--scores', type=Path, nargs='+', required=True)
    args = ap.parse_args()
    if args.output.exists() or args.audit.exists():
        ap.error('Provide fresh output paths')
    reference = load(args.repair)
    prepared = list(cached_queries(args.manifest, args.scores))
    split, = {r['split'] for r, _ in prepared}
    archived = index({'per_query': reference[f'{"oof" if split == "train" else "dev"}_per_query']})
    if archived.keys() != {(r['domain'], r['qid']) for r, _ in prepared}:
        raise ValueError('Archived rankings and manifest cover different queries')
    labels = {d: read_qrels(args.data_root / d / f'qrels_{split}.txt') for d, _ in archived}
    output, audit = [], []
    for r, scores in prepared:
        key = r['domain'], r['qid']
        a = reference['choices'][r['cohort']] if split == 'train' else reference['selected']
        start = archived[key]['rankings'][a]
        if len(set(start)) != 10 or not set(start) <= set(r['docs']):
            raise ValueError(f'Invalid reference for {key}')
        proposals = actions(r['docs'], start, scores)
        output.append({k: r[k] for k in ('domain', 'qid', 'query', 'split', 'cohort', 'actual_reference')} |
                      {'reference': start, 'proposals': proposals})
        gold = labels[key[0]][key[1]]
        value = ndcg10(start, gold)
        oracle = {kind: max([value] + [ndcg10(p['ranking'], gold) for p in proposals if p['type'] == kind])
                  for kind in ('head', 'admission')}
        audit.append({'domain': key[0], 'qid': key[1], 'score': value, 'oracles': oracle |
                      {'any': max(oracle.values())}})
    methods = {k: macro((r['domain'], r['oracles'][k]) for r in audit) for k in ('head', 'admission', 'any')}
    methods['baseline'] = macro((r['domain'], r['score']) for r in audit)
    print(split, methods, 'pairs', sum(len(r['proposals']) for r in output), flush=True)
    for path, payload in ((args.output, {'queries': output}), (args.audit, {'summary': methods, 'per_query': audit})):
        with path.open('x', encoding='utf-8') as file:
            json.dump(payload, file, separators=(',', ':'))
            file.write('\n')


if __name__ == '__main__':
    main()
