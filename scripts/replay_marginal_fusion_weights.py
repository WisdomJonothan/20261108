"""Cross-fit fusion weights from a family's marginal contribution to nDCG."""
import argparse
import json
from itertools import permutations
from pathlib import Path
from statistics import mean

from official.scorer import load_qrels, ndcg_at_k
from replay_consensus_head import head
from replay_dependent_evidence_votes import contribution
from replay_evidence_ranking_consensus import reciprocal
from replay_incumbent_evidence import PANELS, effect, key, macro, read_parts


FAMILIES = (('current',), ('reasoned', 'verified'), ('pointwise',))
ORDERS = tuple(permutations(range(len(FAMILIES))))
FULL = (1 << len(FAMILIES)) - 1


def prepare(archive_dir):
    ranking_archive = archive_dir / 'paired-membership-20261127.tar.gz'
    rows = [row for part in read_parts(ranking_archive, r'paired-membership-.*-20261127')
            for row in part['per_query']]
    records = []
    for panel, release, basename in PANELS:
        archive = archive_dir / f'{release}.tar.gz'
        consensus_name = 'confirmation-consensus' if panel == 'confirm-155' else 'consensus'
        consensus, = read_parts(archive, consensus_name)
        votes = {key(row): row['rankings'] for row in consensus['per_query']}
        evidence = {key(row): row['observations'] for part in read_parts(archive, basename)
                    for row in part['per_query']}
        population = [row for row in rows if row['panel'] == panel]
        if votes.keys() != evidence.keys() or evidence.keys() != {key(r) for r in population}:
            raise ValueError(f'The archived scores do not match the panel {panel}')
        for row in population:
            q = key(row)
            source = votes[q]
            current = row['rankings']['current']
            prior = [source[name] for group in FAMILIES for name in group]
            pointwise = source['pointwise']
            if head(reciprocal(prior, 60), pointwise, evidence[q], True, .5) != current:
                raise ValueError(f'The four archived lists must reproduce {q}')
            records.append({'domain': q[0], 'qid': q[1], 'panel': panel, 'current': current,
                            'pointwise': pointwise, 'proof': evidence[q],
                            'votes': [tuple(contribution(source[name]) for name in group)
                                      for group in FAMILIES]})
    if len(records) != 506 or len({key(r) for r in records}) != 506:
        raise ValueError('Unexpected training-side population')
    return records


def fusion(votes, weights, mask):
    active = [(family, w) for i, (family, w) in enumerate(zip(votes, weights, strict=True))
              if mask & (1 << i)]
    docs = dict.fromkeys(d for family, _ in active for order in family for d in order)
    return sorted(docs, key=lambda d: -sum(
        w * mean(order.get(d, 0.) for order in family) for family, w in active
    ))[:10]


def fitted_weights(values):
    marginal = [[] for _ in FAMILIES]
    for order in ORDERS:
        mask = 0
        for i in order:
            updated = mask | (1 << i)
            marginal[i].append(values[updated] - values[mask])
            mask = updated
    return {'standalone_quality': [values[1 << i] for i in range(len(FAMILIES))],
            'marginal_value': [max(0., mean(delta)) for delta in marginal]}


def cross_fit(records, gold):
    domains = sorted({r['domain'] for r in records})
    groups = {d: [r for r in records if r['domain'] == d] for d in domains}
    raw = {}
    for d, group in groups.items():
        raw[d] = [0.] + [mean(ndcg_at_k(fusion(r['votes'], (1, 1, 1), mask), gold[d][r['qid']])
                                 for r in group) for mask in range(1, FULL+1)]
    predictions, fits = [], {}
    for d, group in groups.items():
        means = [mean(raw[source][mask] for source in domains if source != d) for mask in range(FULL+1)]
        weights = fitted_weights(means)
        fits[d] = weights
        for r in group:
            ranks = {'current': r['current']}
            for name, w in {'uniform_family': (1, 1, 1), **weights}.items():
                ranks[name] = head(fusion(r['votes'], w, FULL), r['pointwise'], r['proof'], True, .5)
            scores = {name: ndcg_at_k(order, gold[d][r['qid']]) for name, order in ranks.items()}
            predictions.append({k: r[k] for k in ('domain','qid','panel')} |
                               {'rankings': ranks, 'scores': scores})
    return predictions, fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-dir', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    records = prepare(args.archive_dir)
    gold = {d: load_qrels(args.data_root / d / 'qrels_train.txt') for d in {r['domain'] for r in records}}
    rows, fits = cross_fit(records, gold)
    contrasts = {f'{a}_vs_{b}': effect(rows, a, b) for a,b in
                 (('uniform_family', 'current'), ('standalone_quality', 'current'),
                  ('marginal_value', 'current'), ('marginal_value', 'standalone_quality'))}
    eligible = all(min(contrasts[f'marginal_value_vs_{b}'][k] for k in
                       ('point', 'minimum_leave_one_domain_out')) > 0
                   for b in ('current', 'standalone_quality'))
    summary = {'n': len(rows), 'families': FAMILIES, 'fit_by_omitted_domain': fits,
               'macro': {name: macro(rows, name) for name in rows[0]['rankings']},
               'contrasts': contrasts, 'eligible': eligible}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'per_query': rows}, handle, separators=(',', ':'))
    print(json.dumps({'macro': summary['macro'], 'eligible': eligible}, indent=2))


if __name__ == '__main__':
    main()
