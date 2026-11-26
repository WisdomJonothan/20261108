"""Isolate the effect of checking the incumbent on a fixed swap proposal."""
import argparse
import json
from pathlib import Path
import re
from statistics import mean
import tarfile

from official.scorer import load_qrels, ndcg_at_k
from replay_anchor_edit_scope import paired_delta


PANELS = (
    ('confirm-155', 'temporal_conflict_disjoint_confirmation_20261126', r'scores-only-confirm-reasoner-p\d'),
    ('reserve-109', 'near-adapter-reserve-20261126', r'old-evidence-p\d'),
    ('reserve-242', 'locked-local-vote-reserve-242-20261126', r'proof-p\d'),
)
ARMS = ('current', 'one_swap', 'entrant_only', 'paired')


def key(row):
    return row['domain'], row['qid']


def read_parts(path, pattern):
    with tarfile.open(path, 'r|gz') as archive:
        for item in archive:
            if re.fullmatch(pattern + r'\.json', item.name.removeprefix('./')):
                yield json.load(archive.extractfile(item))


def grade(evidence):
    return evidence['reasoned'] if evidence['verified'] else evidence['direct']


def decisions(parts, rankings):
    proof = {}
    for part in parts:
        for row in part['per_query']:
            if key(row) in proof:
                raise ValueError('The proof shards overlap')
            proof[key(row)] = row['observations']
    if proof.keys() != {key(r) for r in rankings}:
        raise ValueError('Each training query needs a matching proof row')
    rows = []
    for row in rankings:
        before, after = (row['rankings'][arm] for arm in ARMS[:2])
        pair = row['proposal']
        available = entrant_only = paired = False
        if pair is not None:
            slot, candidate = pair
            if not 1 <= slot <= 9 or candidate in before or after != before[:slot]+[candidate]+before[slot+1:]:
                raise ValueError(f'Invalid fixed swap for {key(row)}')
            incumbent = before[slot]
            evidence = proof[key(row)]
            available = {candidate, incumbent} <= evidence.keys()
            if available:
                new, old = evidence[candidate], evidence[incumbent]
                entrant_only = new['verified'] and new['reasoned'] >= 2
                paired = entrant_only and grade(new) > grade(old)
        elif before != after:
            raise ValueError('A query with no proposal must preserve the original order')
        rows.append({'domain': row['domain'], 'qid': row['qid'], 'panel': row['panel'],
                     'available': available, 'nominated': pair is not None,
                     'prior_scores': {arm: row['scores'][arm] for arm in ARMS[:2]},
                     'rankings': {'current': before, 'one_swap': after,
                                  'entrant_only': after if entrant_only else before,
                                  'paired': after if paired else before}})
    return rows


def macro(rows, arm):
    domains = {r['domain'] for r in rows}
    return mean(mean(r['scores'][arm] for r in rows if r['domain'] == domain)
                for domain in sorted(domains))


def effect(rows, arm, reference):
    result = paired_delta([(r['domain'], r['scores'][arm] - r['scores'][reference]) for r in rows])
    domains = result['per_domain']
    result['minimum_leave_one_domain_out'] = min(
        mean(v for d, v in domains.items() if d != omit) for omit in domains
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-dir', 'data-root', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    args = parser.parse_args()
    previously_ranked = [r for part in read_parts(args.archive_dir / 'paired-membership-20261127.tar.gz',
                                                  r'paired-membership-.*-20261127')
                         for r in part['per_query']]
    rows = []
    for panel, release, stem in PANELS:
        cohort = [r for r in previously_ranked if r['panel'] == panel]
        rows.extend(decisions(read_parts(args.archive_dir / f'{release}.tar.gz', stem), cohort))
    if len(rows) != 506 or len({key(r) for r in rows}) != len(rows):
        raise ValueError('Unexpected or overlapping training panels')
    labels = {d: load_qrels(args.data_root / d / 'qrels_train.txt') for d in {r['domain'] for r in rows}}
    for row in rows:
        d, qid = key(row)
        row['scores'] = {arm: ndcg_at_k(order, labels[d][qid]) for arm, order in row['rankings'].items()}
        if any(abs(row['scores'][arm]-value) > 1e-12 for arm, value in row.pop('prior_scores').items()):
            raise ValueError(f'Archived baseline score differs for {(d, qid)}')
    comparisons = {f'{a}_vs_{b}': effect(rows, a, b) for a, b in
                   (('one_swap', 'current'), ('entrant_only', 'current'),
                    ('paired', 'current'), ('paired', 'entrant_only'))}
    eligible = all(min(comparisons[f'paired_vs_{b}'][m] for m in
                       ('point', 'minimum_leave_one_domain_out')) > 0
                   for b in ('current', 'entrant_only'))
    summary = {
        'queries': len(rows),
        'nominated': sum(r['nominated'] for r in rows),
        'both_graded': sum(r['available'] for r in rows),
        'edits': {a: sum(r['rankings'][a] != r['rankings']['current'] for r in rows)
                  for a in ARMS[1:]},
        'scores': {a: macro(rows, a) for a in ARMS},
        'contrasts': comparisons,
        'paired_eligible': eligible,
    }
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'per_query': rows}, handle, separators=(',', ':'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
