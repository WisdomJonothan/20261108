"""Test one membership change with and without a normalization-stability check."""
import argparse
import json
from pathlib import Path
import re
from statistics import mean
import tarfile

import numpy as np

from audit_rrf_admission import by_topic, load_parts, read_votes
from audit_wide_vote_contributions import member
from official.scorer import load_qrels, ndcg_at_k
from replay_anchor_cap import z
from replay_anchor_edit_scope import paired_delta
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from replay_wide_pointwise_vote import retained_candidate_list


DISCOVERY = (
    ('held', 'wide-vote-held-calibrated.json', 'initial-186'),
    ('confirm', 'wide-vote-confirm-result.json', 'confirm-155'),
    ('reserve', 'wide-vote-reserve-result.json', 'reserve-191'),
)
UNGUARDED, GUARDED = 'one_swap', 'stable_membership_swap'


def anchored_orders(archive, near=False):
    if near:
        # This earlier bundle names the extra 8B and proof shards differently.
        parts = {}
        with tarfile.open(archive, 'r|gz') as stream:
            for entry in stream:
                name = entry.name.removeprefix('./')
                if name in ('consensus.json', 'scopes.json'):
                    parts[name] = json.load(stream.extractfile(entry))
                elif re.fullmatch(r'(extra8b-p\d|old-evidence-p\d|old-pointwise)\.json', name):
                    key = (name.replace('extra8b-', 'pointwise-')
                           .replace('old-evidence-', 'proof-')
                           .replace('old-pointwise.json', 'old-pointwise-p0.json'))
                    parts[key] = json.load(stream.extractfile(entry))
    else:
        parts = load_parts(archive)
    points, proof = read_votes(parts)
    voters = by_topic(parts['consensus.json']['per_query'])
    scopes = by_topic(parts['scopes.json']['queries'])
    if not (voters.keys() == scopes.keys() == points.keys() == proof.keys()):
        raise ValueError('Input archive topics do not align')
    result = {}
    position_prior = z(1 / np.log2(2 + np.arange(30)))
    for key, scope in scopes.items():
        original, wide = scope['original'], scope['wide']
        candidates = retained_candidate_list(original, wide)
        values = np.array([points[key][d] for d in original[:30]])
        incoming = np.array([points[key][d] for d in candidates])
        score = position_prior + (incoming - values.mean()) / max(float(values.std()), 1e-8)
        vote = [candidates[i] for i in np.argsort(-score, kind='stable')[:10]]
        prior = voters[key]['rankings']
        shared = [prior[arm] for arm in ('current', 'reasoned', 'verified')]
        result[key] = (
            head(reciprocal(shared + [prior['pointwise']], 60), prior['pointwise'], proof[key], True, .5),
            head(reciprocal(shared + [vote], 60), prior['pointwise'], proof[key], True, .5),
        )
    return result


def nominate(current, alternative):
    entering = next((d for d in alternative if d not in current), None)
    # Replace the deepest displaced document, keeping rank one and every other position.
    slot = next((i for i in range(9, 0, -1) if current[i] not in alternative), None)
    return (slot, entering) if slot is not None and entering is not None else None


def measured(rows, anchored, data):
    if ((anchored is not None and len(rows) != len(anchored))
            or any(row['split'] != 'train' for row in rows)):
        raise ValueError('Expected aligned training-side topics')
    qrels = {d: load_qrels(data / d / 'qrels_train.txt') for d in {r['domain'] for r in rows}}
    per_query = []
    for row in rows:
        domain, qid = row['domain'], row['qid']
        before = row['rankings']['current']
        after = row['rankings']['local_candidate_scale']
        if anchored is not None:
            baseline, alternate = anchored[domain, qid]
            if before != baseline:
                raise ValueError(f'Baseline changed for {domain}/{qid}')
        gold = qrels[domain][qid]
        proposal = nominate(before, after)
        first = before.copy()
        stable = False if anchored is not None else None
        outcome = None
        if proposal is not None:
            slot, entering = proposal
            displaced = before[slot]
            first[slot] = entering
            if anchored is not None:
                stable = entering in alternate and displaced not in alternate
            outcome = int(entering in gold) - int(displaced in gold)
        ranks = {'current': before, UNGUARDED: first}
        if anchored is not None:
            ranks.update(local_candidate_scale=after, stable_membership_swap=first if stable else before)
        scores = {name: ndcg_at_k(order, gold) for name, order in ranks.items()}
        if abs(scores['current'] - row['scores']['current']) > 1e-12:
            raise ValueError(f'Baseline score mismatch for {domain}/{qid}')
        per_query.append({'domain': domain, 'qid': qid, 'panel': row['panel'],
                          'rankings': ranks, 'proposal': proposal, 'stable': stable,
                          'pair_outcome': outcome, 'scores': scores})
    return per_query


def summarize(rows):
    def compare(arm, baseline):
        delta = paired_delta([(r['domain'], r['scores'][arm] - r['scores'][baseline]) for r in rows])
        by_domain = delta['per_domain']
        delta['minimum_leave_one_domain_out'] = min(
            mean(v for d, v in by_domain.items() if d != omitted) for omitted in by_domain
        )
        return delta

    domains = {r['domain'] for r in rows}
    arms = rows[0]['scores']
    summary = {
        'n': len(rows),
        'score': {arm: mean(mean(r['scores'][arm] for r in rows if r['domain'] == d) for d in domains)
                  for arm in arms},
        'versus_frozen': {arm: compare(arm, 'current') for arm in arms if arm != 'current'},
    }
    if GUARDED in arms:
        summary['guard_versus_same_proposal'] = compare(GUARDED, UNGUARDED)
    outcomes = ('all', None), ('accepted', True), ('blocked', False)
    summary['proposals'] = {
        key: {name: sum((stable is None or r['stable'] == stable) and r['pair_outcome'] == value
                        for r in rows)
              for name, value in (('helps', 1), ('ties', 0), ('harms', -1))}
        for key, stable in (outcomes if GUARDED in arms else outcomes[:1])
    }
    return summary


def panel_omission_deltas(rows, arm):
    panels = sorted({r['panel'] for r in rows})
    return {omitted: paired_delta([
        (r['domain'], r['scores'][arm] - r['scores']['current'])
        for r in rows if r['panel'] != omitted
    ])['point'] for omitted in panels}


def run(analysis, data, confirmation, policy):
    def pack(name):
        return analysis / f'{name}-20261126.tar.gz'

    if confirmation is None:
        rows = member(pack('separated-gain-crossfit'), 'separated-gain-training-result.json')['per_query']
        anchored = {}
        archived = by_topic(rows)
        for name, filename, panel in DISCOVERY:
            for r in member(pack(f'wide-vote-{name}'), filename)['per_query']:
                key = r['domain'], r['qid']
                if archived[key]['panel'] != panel or key in anchored:
                    raise ValueError(f'Unexpected archived panel for {key}')
                ranks = r['rankings']
                anchored[key] = ranks['current'], ranks['original_candidate_scale']
        anchored.update(anchored_orders(pack('near-adapter-reserve'), near=True))
        if len(rows) != 641:
            raise ValueError('Unexpected discovery population')
    else:
        prior = json.loads(confirmation.read_text(encoding='utf-8'))
        if prior['stage'] != 'discovery':
            raise ValueError('Expected the archived discovery result')
        if policy == GUARDED and not prior['eligible']:
            raise ValueError('The guarded comparison failed its discovery gate')
        if policy == UNGUARDED:
            metrics = prior['summary']['versus_frozen'][UNGUARDED]
            if min(metrics['point'], metrics['minimum_leave_one_domain_out'],
                   *panel_omission_deltas(prior['per_query'], UNGUARDED).values()) <= 0:
                raise ValueError('The unguarded follow-up requires positive domain and panel omission checks')
        archive = pack('locked-local-vote-reserve-242')
        rows = [r | {'panel': 'reserve-242'} for r in member(archive, 'result.json')['per_query']]
        anchored = anchored_orders(archive) if policy == GUARDED else None
        if len(rows) != 242 or set(by_topic(rows)) & set(by_topic(prior['per_query'])):
            raise ValueError('Expected the disjoint 242-topic confirmation panel')
    graded = measured(rows, anchored, data)
    summary = summarize(graded)
    change = summary['versus_frozen'][policy]
    eligible = min(change['point'], change['minimum_leave_one_domain_out']) > 0
    if policy == GUARDED:
        eligible = eligible and summary['guard_versus_same_proposal']['point'] > 0
    return {'stage': 'discovery' if confirmation is None else 'confirmation',
            'policy': policy, 'eligible': eligible, 'summary': summary, 'per_query': graded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-dir', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, help='A passing frozen discovery result')
    parser.add_argument('--confirmation-policy', choices=(GUARDED, UNGUARDED), default=GUARDED)
    args = parser.parse_args()
    if args.confirmation is None and args.confirmation_policy != GUARDED:
        parser.error('The separate one-swap follow-up requires a discovery record')
    result = run(args.archive_dir, args.data_root, args.confirmation, args.confirmation_policy)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, separators=(',', ':'))
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
