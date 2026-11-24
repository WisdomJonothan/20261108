"""Replay an anchor budget of 19 document pairs while preserving the original top ten."""
import argparse
import json
from pathlib import Path

import numpy as np

from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_position_discount import bootstrap


def z(scores):
    scores = np.asarray(scores)
    return (scores - scores.mean()) / max(float(scores.std()), 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding='utf-8'))
    result = []
    for domain in sorted({r['domain'] for r in source['per_query']}):
        rows = [r for r in source['per_query'] if r['domain'] == domain]
        directory = args.root / 'runs'
        dense = read_trec_run(directory / f'dense_mxbai_{domain}_train.trec')
        fused = {f: read_trec_run(directory / f'trainfuse_{f}_{domain}_train.trec')
                 for f in {r['fold'] for r in rows}}
        gold = read_qrels(args.root / 'data/track1_tempo' / domain / 'qrels_train.txt')
        for item in rows:
            qid = item['qid']
            top = item['arms']['base']['ranking']
            original = [doc for _, _, doc in dense[qid][:100]]
            selected = set(list(dict.fromkeys(top + original))[:20])
            pool = [doc for doc in original if doc in selected]
            base = {doc: score for _, score, doc in fused[item['fold']][qid] if doc in selected}
            assert len(base) == len(pool) == 20 and item['anchor'] in selected
            value = .7 * z([base[doc] for doc in pool]) + .3 * z([item['scores'][doc] for doc in pool])
            ranking = [pool[i] for i in np.argsort(-value, kind='stable')[:10]]
            original_ndcg = item['arms']['base']['ndcg']
            relevant = gold[qid]
            assert abs(ndcg10(top, relevant) - original_ndcg) < 1e-10
            result.append({'domain': domain, 'qid': qid, 'base': original_ndcg,
                           'full': item['arms']['base_plus_anchor30']['ndcg'],
                           'cap20': ndcg10(ranking, relevant), 'ranking': ranking})
    assert len(result) == source['queries']
    report = {'scope': 'Exploratory replay of the frozen top ten plus ten highest dense outsiders. '
              'Use the same anchor and 70:30 normalized fusion. The new ranking uses 19 '
              'bidirectional comparisons per query if executed directly. This replay scores '
              'only documents available in the existing full cache.',
              'queries': len(result), 'prompts_per_query': 38,
              'scores': {name: bootstrap([(r['domain'], r[name]) for r in result])['point']
                         for name in ('base', 'full', 'cap20')},
              'versus_original': bootstrap([(r['domain'], r['cap20'] - r['base']) for r in result]),
              'versus_full': bootstrap([(r['domain'], r['cap20'] - r['full']) for r in result]),
              'per_query': result}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print({k: v for k, v in report.items() if k != 'per_query'})


if __name__ == '__main__':
    main()
