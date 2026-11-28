"""Test grounded year checks for a rank-ten edit on fixed training panels."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import isfinite
from pathlib import Path
import re

from audit_wide_vote_contributions import member
from benchmark_pairwise_replacement_v2 import strip_html, terms
from coverage_cf_bound import read_steps
from eval_contrastive_evidence_retrieval import interleave
from eval_event_date_windows import write_gz
from eval_fifth_ranker import prepare
from official.scorer import ndcg_at_k
from replay_conditional_rank_fusion import key, labels, measure
from replay_contrastive_evidence_admission import read
from replay_incumbent_evidence import effect, macro
from replay_list_conditioned_admission import choose


PANELS = {
    'held': (186, 'wide-vote-held-calibrated.json'),
    'confirm': (155, 'wide-vote-confirm-result.json'),
    'reserve': (191, 'wide-vote-reserve-result.json'),
}
ARMS = ('current', 'focused', 'score_head', 'any_year', 'linked_pair', 'linked_head')
WORD = r'(?:1\d{3}|20[0-2]\d)'
YEAR = re.compile(rf'(?<![\w/-]){WORD}(?![\w/-]|[\x27\u2019]s)', re.I)
CONDITION = re.compile(
    rf'\b(?:(between|from)\s+({WORD})\s+(?:and|to|through)\s+({WORD})'
    rf'|(before|prior to|after|since|by|in|during)\s+({WORD}))'
    r'(?![\w/-]|[\x27\u2019]s)', re.I
)
SENTENCE = re.compile(r'(?<=[.!?])\s+|[\r\n]+|\s*\[Additional excerpt\]\s*')
GENERIC = set('before prior after since during between through and which passage reports documents '
              'evidence know source sources there information find answers answer time'.split())


def years(text):
    return {int(year) for year in YEAR.findall(text)}


def query_spec(text):
    hits = []
    for sentence in SENTENCE.split(strip_html(text)):
        if not sentence.rstrip().endswith('?'):
            continue
        matches = list(CONDITION.finditer(sentence))
        if len(matches) != 1:
            continue
        match = matches[0]
        if match[1]:
            lo, hi = int(match[2]), int(match[3])
        else:
            year = int(match[5])
            lo, hi = {
                'before': (1000, year - 1), 'prior to': (1000, year - 1),
                'after': (year + 1, 2029), 'since': (year, 2029),
                'by': (1000, year), 'in': (year, year), 'during': (year, year),
            }[match[4].lower()]
        words = sorted(terms(YEAR.sub(' ', sentence)) - GENERIC)
        if lo <= hi and len(words) >= 2:
            hits.append({'query': ' '.join(sentence.split()), 'lo': lo, 'hi': hi, 'words': words,
                         'condition': match[0].lower()})
    return hits[0] if len(hits) == 1 else None


def evidence_state(excerpt, spec):
    sentences = SENTENCE.split(excerpt)
    vocabulary = set(spec['words'])
    overlaps = [len(vocabulary & terms(YEAR.sub(' ', line))) for line in sentences]
    strength = max(overlaps)
    best = [line for line, overlap in zip(sentences, overlaps, strict=True) if overlap == strength]
    dated = [years(line) for line in best]
    # A tie with undated evidence, several candidate years, or weak localization is unresolved.
    common = set.intersection(*dated) if strength >= 2 and all(len(y) == 1 for y in dated) else set()
    linked = next(iter(common), None)

    def inside(value):
        return spec['lo'] <= value <= spec['hi']

    return {'year': linked, 'overlap': strength, 'any_year': any(map(inside, years(excerpt))),
            'linked': None if linked is None else inside(linked),
            'witness': best[0] if linked is not None else None}


def build(args):
    cases = [row for file in args.candidates for row in read(file)['per_query']]
    if Counter(row['panel'] for row in cases) != {p: count for p, (count, _) in PANELS.items()}:
        raise ValueError('Expected the complete 532-question training population')
    if len({key(row) for row in cases}) != len(cases):
        raise ValueError('Training panels overlap')
    archives = {}
    for panel, (_, name) in PANELS.items():
        path = args.archive_dir / f'wide-vote-{panel}-20261126.tar.gz'
        archives.update({key(row): row for row in member(path, name)['per_query']})
    if archives.keys() != {key(row) for row in cases}:
        raise ValueError('Expected the same archived rankings')
    grouped = defaultdict(list)
    for row in cases:
        grouped[row['domain']].append(row)
    inputs, fixed = [], []
    for domain, rows in sorted(grouped.items()):
        with (args.data_root / domain / 'examples_train.jsonl').open(encoding='utf-8') as stream:
            queries = {r['id']: r['query'] for r in map(json.loads, stream)}
        for case in rows:
            current = case['current']
            archived = archives[key(case)]
            if current != archived['rankings']['current'] or archived['split'] != 'train':
                raise ValueError(f'The frozen training ranking changed on {key(case)}')
            if case['panel'] == 'reserve':
                offers = case['offers']['two_sided']
            else:
                offers = interleave(case['offers']['contrast'], case['offers']['reversal'])
            docs = current + offers
            if len(current) != 10 or len(docs) != 14 or len(set(docs)) != 14:
                raise ValueError(f'Expected fourteen different documents for {key(case)}')
            row = {name: case[name] for name in ('domain', 'qid', 'panel', 'current')} | {
                'prior_score': archived['scores']['current'], 'docs': docs}
            spec = query_spec(queries[row['qid']])
            fixed.append(row | {'spec': spec})
            if spec:
                inputs.append(row | {'sources': [offers]})
    args.official = False
    packed = prepare(args, inputs)['per_query']
    selected = {key(r): r['spec'] for r in fixed if r['spec']}
    for row in packed:
        spec = selected[key(row)]
        full = row['query']
        # The complete question selects the excerpt; the dated request alone is scored.
        row['query'] = spec['query']
        if not full or not row['query']:
            raise ValueError(f'Missing text on {key(row)}')
        row['evidence'] = [evidence_state(excerpt, spec) for excerpt in row['excerpts']]
    signature = fingerprint(packed)
    print('active:', dict(Counter(r['panel'] for r in packed)), 'input_sha256:', signature, flush=True)
    write_gz(args.score_inputs, {'per_query': packed})
    return {'input_sha256': signature, 'per_query': fixed}


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def rank(fixed, scored):
    expected = {key(row) for row in fixed if row['spec']}
    by_key = {key(row): row for row in scored}
    if by_key.keys() != expected or len(by_key) != len(scored):
        raise ValueError('Score coverage differs from the frozen trigger')
    for row in fixed:
        current, docs = row['current'], row['docs']
        orders = dict.fromkeys(ARMS, current)
        visible = by_key.get(key(row))
        if visible:
            if visible['docs'] != docs or visible['current'] != current:
                raise ValueError(f'Candidate order changed on {key(row)}')
            raw = visible['new_scores']
            if len(raw) != len(docs) or not all(map(isfinite, raw)):
                raise ValueError(f'Invalid score vector on {key(row)}')
            scores = dict(zip(docs, raw, strict=True))
            state = dict(zip(docs, visible['evidence'], strict=True))
            offers = docs[10:]
            incumbent = current[9]
            orders['focused'] = choose(current, offers, {d: scores[d] - scores[incumbent] for d in offers})
            head_floor = max(scores[d] for d in current)
            orders['score_head'] = choose(current, offers, {d: scores[d] - head_floor for d in offers})
            for arm, field, against in (('any_year', 'any_year', current[:9]),
                                        ('linked_pair', 'linked', []),
                                        ('linked_head', 'linked', current[:9])):
                floor = max([scores[incumbent], *(scores[d] for d in against if state[d][field] is True)])
                allowed = [d for d in offers if state[d][field] is True]
                if allowed:
                    orders[arm] = choose(current, allowed, {d: scores[d] - floor for d in allowed})
        yield row | {'rankings': orders}


def summarize(rows, steps, gold):
    contrasts = {a: effect(rows, 'linked_head', a) for a in ARMS if a != 'linked_head'}
    return {'queries': len(rows), 'domains': len({r['domain'] for r in rows}),
            'triggered': sum(r['spec'] is not None for r in rows),
            'scores': {a: macro(rows, a) for a in ARMS},
            'oracle': ceilings(rows, gold),
            'outcomes': {a: dict(step_counts(rows, steps, a)) for a in ARMS},
            'contrasts': contrasts}


def ceilings(rows, gold):
    scored = []
    improved = Counter(all=0, triggered=0)
    for row in rows:
        relevant = gold[key(row)]
        current, docs = row['current'], row['docs']
        head, incumbent, offers = current[:9], current[9], docs[10:]
        best = next((doc for doc in offers if doc in relevant), incumbent)
        rank_ten = current if incumbent in relevant else head + [best]
        improved['all'] += rank_ten != current
        improved['triggered'] += rank_ten != current and row['spec'] is not None
        orders = {'rank_ten': rank_ten, 'triggered_rank_ten': rank_ten if row['spec'] else current,
                  'same_ten_reorder': sorted(current, key=lambda doc: doc not in relevant),
                  'free_reorder_14': sorted(docs, key=lambda doc: doc not in relevant)}
        scored.append({'domain': row['domain'],
                       'scores': {a: ndcg_at_k(docs, relevant) for a, docs in orders.items()}})
    return {'scores': {a: macro(scored, a) for a in scored[0]['scores']},
            'useful_swap_queries': improved}


def publish_evidence(evidence):
    display = evidence.copy()
    quote = display.pop('witness')
    display['witness_sha256'] = hashlib.sha256(quote.encode()).hexdigest() if quote else None
    return display


def step_counts(rows, steps, arm):
    total = Counter(changed=0, improved=0, declined=0, gained_steps=0, lost_steps=0)
    for row in rows:
        before, after = (row['rankings'][name] for name in ('current', arm))
        gold = steps[row['domain']].get(row['qid'], {})
        prior = {s for s, docs in gold.items() if not docs.isdisjoint(before)}
        new = {s for s, docs in gold.items() if not docs.isdisjoint(after)}
        delta = row['scores'][arm] - row['scores']['current']
        total.update(changed=before != after, improved=delta > 1e-12, declined=delta < -1e-12,
                     gained_steps=len(new - prior), lost_steps=len(prior - new))
    return total


def evaluate(args):
    fixed, scored = read(args.fixed), read(args.scores)
    observed = sorted(scored['per_query'], key=key)
    prepared = read(args.score_inputs)['per_query']
    if fixed['input_sha256'] != fingerprint(prepared):
        raise ValueError('Input manifest changed after preparation')
    for a, b in zip(observed, sorted(prepared, key=key), strict=True):
        if key(a) != key(b) or a['evidence'] != b['evidence']:
            raise ValueError(f'Witnesses changed on {key(a)}')
    ranked = list(rank(fixed['per_query'], observed))
    locked = fingerprint(ranked)
    gold = labels(ranked, args.data_root)
    rows = measure(ranked, gold)
    steps = {d: read_steps(args.data_root / d / 'qrels_steps_train.txt') for d in {r['domain'] for r in rows}}
    panels = {'all': summarize(rows, steps, gold)}
    panels.update({p: summarize([r for r in rows if r['panel'] == p], steps, gold) for p in PANELS})
    for row in rows:
        if row['spec']:
            row['spec'] = {name: row['spec'][name] for name in ('lo', 'hi', 'condition')}
    for row in observed:
        row['evidence'] = [publish_evidence(state) for state in row['evidence']]
    gate = all(p['contrasts'][a]['minimum_leave_one_domain_out'] > 0 and p['contrasts'][a]['point'] > 0
               for p in panels.values() for a in ('current', 'any_year', 'focused'))
    result = {'model': scored['model'], 'input_sha256': fixed['input_sha256'], 'rank_fingerprint': locked,
              'inference_seconds': scored['inference_seconds'], 'eligible': gate,
              'panels': panels, 'per_query': rows, 'raw_scores': observed}
    print(json.dumps({n: {k: v for k, v in p.items() if k != 'contrasts'} for n, p in panels.items()}, indent=2))
    return result


def diagnostic(args):
    from compare_anchor_reference import load
    from eval_temporal_query_contrast import SIDES, prepare as prepare_pairs

    examples = prepare_pairs(load(args.manifest)['cases'], args.data_root)
    fields = ('any_year', 'linked')
    totals = {field: Counter(correct=0, wrong=0, abstain=0, correct_flips=0) for field in fields}
    rows = []
    for case, queries, passages in examples:
        trials = {}
        for side, query in queries.items():
            spec = query_spec(query)
            evidence = {d: evidence_state(passages[d], spec) for d in SIDES} if spec else {}
            outcomes = {}
            for field in fields:
                supported = [d for d, state in evidence.items() if state[field] is True]
                result = 'correct' if supported == [side] else 'wrong' if len(supported) == 1 else 'abstain'
                totals[field][result] += 1
                outcomes[field] = result
            trials[side] = {'query': query, 'spec': spec, 'outcomes': outcomes,
                            'evidence': {d: publish_evidence(state) for d, state in evidence.items()}}
        for field in fields:
            totals[field]['correct_flips'] += all(t['outcomes'][field] == 'correct' for t in trials.values())
        rows.append({'case': case['id'], 'domain': case['domain'], 'trials': trials})
    print(json.dumps(totals, indent=2))
    return {'source_manifest': args.manifest.name, 'cases': len(rows),
            'outcomes': totals, 'per_case': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    prep = stages.add_parser('prepare')
    prep.add_argument('--candidates', nargs=2, type=Path, required=True)
    prep.add_argument('--archive-dir', type=Path, required=True)
    evaluate_stage = stages.add_parser('evaluate')
    evaluate_stage.add_argument('--fixed', type=Path, required=True)
    evaluate_stage.add_argument('--scores', type=Path, required=True)
    probe = stages.add_parser('diagnostic')
    probe.add_argument('--manifest', type=Path, required=True)
    for stage in (prep, evaluate_stage):
        stage.add_argument('--score-inputs', type=Path, required=True)
    for stage in (prep, evaluate_stage, probe):
        for name in ('data-root', 'output'):
            stage.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.stage == 'prepare' and args.score_inputs.exists():
        parser.error('Use new output paths.')
    result = {'prepare': build, 'evaluate': evaluate, 'diagnostic': diagnostic}[args.stage](args)
    write_gz(args.output, result)


if __name__ == '__main__':
    main()
