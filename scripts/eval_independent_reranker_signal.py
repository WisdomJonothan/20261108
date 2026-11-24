"""Audit whether the cached independent reranker adds a transferable direction."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from coverage_cf_bound import DOMAINS, ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_position_discount import bootstrap


def z(values):
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--frozen-run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = json.loads((args.root / 'results/analysis/position_oracle_dev_20261124.json')
                      .read_text(encoding='utf-8'))['per_query']
    decisions = {(r['domain'], r['qid']): r for r in base}
    results = []
    for domain in DOMAINS:
        dense = read_trec_run(args.root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        other = read_trec_run(args.root / 'runs' / f'rerank_qwen3_{domain}_dev.trec')
        frozen = read_trec_run(args.frozen_run_root / f'sub_sft_e2bc_{domain}_dev.trec')
        gold = read_qrels(args.data_root / domain / 'qrels_dev.txt')
        assert set(dense) == set(other) == set(frozen) == set(gold)
        for qid, rows in dense.items():
            pool = [doc for _, _, doc in rows[:100]]
            old = {doc: score for _, score, doc in frozen[qid] if doc in pool}
            qwen = {doc: score for _, score, doc in other[qid]}
            assert set(pool) == set(qwen) == set(old)
            top = [doc for _, _, doc in frozen[qid][:10]]
            relevant = gold[qid]
            reference = decisions[domain, qid]
            assert abs(ndcg10(top, relevant) - reference['baseline']) < 1e-10
            merged = .75 * z(np.array([old[d] for d in pool])) + .25 * z(np.array([qwen[d] for d in pool]))
            blend = [pool[i] for i in np.argsort(-merged, kind='stable')[:10]]
            selected = reference['policies']['external6']['chosen']
            delta = reference['policies']['external6']['delta']
            veto = bool(selected and qwen[selected['challenger']] < qwen[selected['incumbent']])
            record = {'domain': domain, 'qid': qid, 'base': reference['baseline'],
                      'fixed': reference['baseline'] + delta,
                      'qwen25_blend': ndcg10(blend, relevant),
                      'fixed_with_qwen_veto': reference['baseline'] + (0. if veto else delta),
                      'veto': veto, 'vetoed_delta': delta if veto else 0.,
                      'blend_relevant_enter': len((set(blend) - set(top)) & relevant),
                      'blend_relevant_leave': len((set(top) - set(blend)) & relevant)}
            results.append(record)
    assert len(results) == 519
    summary = {arm: bootstrap([(r['domain'], r[arm]) for r in results])['point']
               for arm in ('base', 'fixed', 'qwen25_blend', 'fixed_with_qwen_veto')}
    summary['blend_vs_base'] = bootstrap([(r['domain'], r['qwen25_blend'] - r['base']) for r in results])
    summary['veto_vs_fixed'] = bootstrap([(r['domain'], r['fixed_with_qwen_veto'] - r['fixed'])
                                         for r in results])
    summary['vetoed_swaps'] = Counter('helpful' if r['vetoed_delta'] > 1e-12
                                     else 'harmful' if r['vetoed_delta'] < -1e-12 else 'neutral'
                                     for r in results if r['veto'])
    report = {'scope': 'Exploratory fixed-weight dev replay. The cached Qwen3-Reranker-0.6B '
              'scores use the existing official template on the dense top100. Mix 25% standardized '
              'Qwen score with the standardized frozen fused score, or veto an already chosen '
              'six-pair replacement when the independent reranker ranks the incumbent higher. '
              'These are distinct controls. No weight or threshold fitting.',
              'summary': summary, 'per_query': results}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
