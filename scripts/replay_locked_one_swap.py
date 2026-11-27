"""Replay the training-selected single swap with a frozen official ranking manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_wide_vote_contributions import member
from official.scorer import load_qrels, ndcg_at_k
from replay_incumbent_evidence import effect, macro
from replay_paired_membership import nominate


REFERENCE = 0.40674178096988606


def write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')


def prepare(args):
    selection = member(args.training, 'paired-membership-242-20261127.json')
    development = member(args.training, 'paired-membership-641-final-20261127.json')
    for summary in (selection['summary'], development['summary']):
        train = summary['versus_frozen']['one_swap']
        if min(train['point'], train['minimum_leave_one_domain_out']) <= 0:
            raise ValueError('Expected positive training and domain-omission estimates')
    if selection['policy'] != 'one_swap' or not selection['eligible']:
        raise ValueError('Expected the locked single-swap confirmation')
    source = member(args.archive, 'result.json')['per_query']
    keys = {(r['domain'], r['qid']) for r in source}
    if len(source) != 519 or len(keys) != 519 or len({d for d, _ in keys}) != 13:
        raise ValueError('Expected the original 519 official topics')
    rows = []
    for row in source:
        if row['split'] != 'dev':
            raise ValueError('Expected official development rankings')
        current = row['rankings']['current']
        proposal = row['rankings']['local_candidate_scale']
        if len(set(current)) != 10 or len(set(proposal)) != 10:
            raise ValueError('Expected ten unique documents in each list')
        edit = nominate(current, proposal)
        after = current.copy()
        if edit:
            slot, document = edit
            after[slot] = document
        rows.append({'domain': row['domain'], 'qid': row['qid'], 'edit': edit,
                     'rankings': {'current': current, 'one_swap': after}})
    write(args.output, {'policy': 'one_swap', 'source': args.archive.name, 'per_query': rows})
    print(hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)


def score(args):
    payload = args.rankings.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.fingerprint:
        raise ValueError('Ranking fingerprint differs from the prepared output')
    manifest = json.loads(payload)
    rows = manifest['per_query']
    domains = {r['domain'] for r in rows}
    if len(rows) != 519 or len(domains) != 13 or manifest['policy'] != 'one_swap':
        raise ValueError('Invalid locked official population')
    gold = {d: load_qrels(args.data_root / d / 'qrels_dev.txt') for d in domains}
    for row in rows:
        relevance = gold[row['domain']][row['qid']]
        row['scores'] = {arm: ndcg_at_k(order, relevance) for arm, order in row['rankings'].items()}
    estimates = {arm: macro(rows, arm) for arm in ('current', 'one_swap')}
    if abs(estimates['current'] - REFERENCE) > 1e-12:
        raise ValueError('Official reference score differs from the published value')
    delta = effect(rows, 'one_swap', 'current')
    summary = {'queries': len(rows), 'domains': len(domains), 'swap_count': sum(bool(r['edit']) for r in rows),
               'scores': estimates, 'change': delta,
               'eligible': min(delta['point'], delta['minimum_leave_one_domain_out']) > 0}
    write(args.output, {'locked_rank_sha256': digest, 'summary': summary, 'per_query': rows})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='stage', required=True)
    prep = subparsers.add_parser('prepare')
    prep.add_argument('--archive', type=Path, required=True)
    prep.add_argument('--training', type=Path, required=True)
    metric = subparsers.add_parser('score')
    metric.add_argument('--rankings', type=Path, required=True)
    metric.add_argument('--fingerprint', required=True)
    metric.add_argument('--data-root', type=Path, required=True)
    for command in (prep, metric):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
