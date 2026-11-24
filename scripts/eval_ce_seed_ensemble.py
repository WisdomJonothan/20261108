"""Evaluate fixed score averages as a stronger starting ranking for replacement."""
import argparse
import json
from pathlib import Path

import numpy as np

from coverage_cf_bound import DOMAINS, ndcg10, read_qrels
from lib_reteco import read_trec_run


SEEDS = ('s17_20261115', 's18_20261118', 's19_20261118')
ARMS = ('full3_equal', 'full3_ce70', 'title3_equal', 'two_views_equal')


def z(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def write_ranking(folder, domain, ranking):
    path = folder / f'full3ce70_20261124_{domain}_dev.trec'
    with path.open('x', encoding='utf-8') as stream:
        for qid, top in ranking.items():
            for rank, (doc, score) in enumerate(top, 1):
                stream.write(f'{qid} Q0 {doc} {rank} {score:.10f} full3ce70_20261124\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--external-replay', type=Path,
                        help='Also replay the recorded external-six decisions on full3_ce70.')
    parser.add_argument('--export-runs', type=Path,
                        help='Write the fixed 0.7 full-query ensemble to a new run prefix.')
    args = parser.parse_args()
    data_root = args.data_root or args.root / 'data/track1_tempo'
    records = []
    replay = {}
    if args.external_replay:
        source = json.loads(args.external_replay.read_text(encoding='utf-8'))
        replay = {(r['domain'], r['qid']): r['policies']['external6']
                  for r in source['per_query']}
        assert len(replay) == 519
    for domain in DOMAINS:
        dense = read_trec_run(args.root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        baseline_path = args.root / 'runs' / f'sub_sft_e2bc_{domain}_dev_head10.trec'
        if not baseline_path.exists():
            baseline_path = args.root / 'runs' / f'sub_sft_e2bc_{domain}_dev.trec'
        baseline = read_trec_run(baseline_path)
        models = {view: [read_trec_run(args.root / 'runs' /
                                      f'fusion100_{view}_ce_{seed}_{domain}_dev_ce.trec')
                         for seed in SEEDS] for view in ('full', 'title')}
        gold = read_qrels(data_root / domain / 'qrels_dev.txt')
        assert set(dense) == set(gold) == set(baseline)
        ranking = {}
        for qid, rows in dense.items():
            ids = [doc for _, _, doc in rows[:100]]
            d = z([s for _, s, _ in rows[:100]])
            means = {}
            for view, runs in models.items():
                vectors = []
                for run in runs:
                    scores = {doc: score for _, score, doc in run[qid]}
                    assert set(scores) == set(ids) and len(scores) == 100
                    vectors.append(z([scores[doc] for doc in ids]))
                means[view] = z(np.mean(vectors, axis=0))
            f, t = means['full'], means['title']
            vectors = dict(full3_equal=d + f, full3_ce70=.3 * d + .7 * f,
                           title3_equal=d + t, two_views_equal=d + z((f + t) / 2))
            row = {'domain': domain, 'qid': qid,
                   'baseline': ndcg10([doc for _, _, doc in baseline[qid]], gold[qid])}
            for arm, vector in vectors.items():
                order = np.argsort(-vector, kind='stable')
                top10 = [ids[i] for i in order[:10]]
                row[arm] = ndcg10(top10, gold[qid])
                if args.export_runs and arm == 'full3_ce70':
                    ranking[qid] = [(ids[i], vector[i]) for i in order]
                if replay and arm == 'full3_ce70':
                    decision = replay[domain, qid]
                    row['external6'] = row['baseline'] + decision['delta']
                    choice = decision['chosen']
                    applied = (choice is not None and choice['incumbent'] in top10
                               and choice['challenger'] not in top10)
                    if applied:
                        top10[top10.index(choice['incumbent'])] = choice['challenger']
                    elif choice:
                        row['old_incumbent_missing'] = choice['incumbent'] not in top10
                        row['challenger_already_present'] = choice['challenger'] in top10
                    row['full3_ce70_replay'] = ndcg10(top10, gold[qid])
                    row['transferred_action_applied'] = applied
            records.append(row)
        if args.export_runs:
            write_ranking(args.export_runs, domain, ranking)
    assert len(records) == 519
    arms = ('baseline', *ARMS)
    if replay:
        arms += ('external6', 'full3_ce70_replay')
    scores = {arm: np.array([np.mean([r[arm] for r in records if r['domain'] == d]) for d in DOMAINS])
              for arm in arms}
    assert abs(float(scores['baseline'].mean()) - .35809852326897723) < 1e-9
    rng = np.random.default_rng(20261124)
    draws = rng.integers(len(DOMAINS), size=(10000, len(DOMAINS)))
    summary = {}
    for arm, means in scores.items():
        delta = means - scores['baseline']
        summary[arm] = {'ndcg': float(means.mean()), 'delta': float(delta.mean()),
                        'domain_bootstrap_95': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                        'per_domain': dict(zip(DOMAINS, means.tolist()))}
    result = {'protocol': 'Exploratory official dev fixed-weight ensemble. Every CE is z-normalized '
              'on the same 100 candidates, averaged within view and re-standardized. The equal '
              'arms weight CE and dense equally. CE=0.7 transfers the prior seed-17 internal choice. '
              'No fitting or weights chosen on official dev. Standalone ensembles use three or six '
              'CE passes per query in deployment. The inputs reuse cached scores.',
              'seeds': SEEDS, 'queries': 519, 'summary': summary, 'per_query': records}
    if replay:
        assert abs(summary['external6']['ndcg'] - .3735032332323295) < 1e-9
        result['transfer_protocol'] = ('Transfer the original external-six decision to the '
                                       '0.7 full-query ensemble when the same incumbent remains '
                                       'in its top ten and the original challenger is outside. '
                                       'Apply the swap at the incumbent’s current rank.')
        result['transferred_actions'] = sum(r['transferred_action_applied'] for r in records)
        result['original_selected_actions'] = sum(r['chosen'] is not None for r in replay.values())
        result['blocked_by_incumbent_missing'] = sum(r.get('old_incumbent_missing', False) for r in records)
        result['blocked_by_challenger_present'] = sum(r.get('challenger_already_present', False) for r in records)
        delta = scores['full3_ce70_replay'] - scores['external6']
        result['transfer_vs_external6'] = {
            'delta': float(delta.mean()),
            'domain_bootstrap_95': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print({arm: {k: v for k, v in values.items() if k != 'per_domain'}
           for arm, values in summary.items()})
    if replay:
        print('Transferred actions:', result['transferred_actions'],
              'versus external-six:', result['transfer_vs_external6'])


if __name__ == '__main__':
    main()
