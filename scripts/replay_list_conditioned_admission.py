"""Compare gap-targeted replacement with and without the retained evidence."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from coverage_cf_bound import read_steps
from eval_contrastive_evidence_retrieval import identify_fragments
from eval_event_date_windows import write_gz
from eval_fifth_ranker import prepare
from lib_reteco import load_examples
from replay_conditional_rank_fusion import key, labels, measure
from replay_contrastive_evidence_admission import read
from replay_incumbent_evidence import effect, macro


ARMS = ('whole', 'gap_only', 'pair', 'list_conditioned')
VIEWS = ('gap', 'covered')


def build(args):
    cases = read(args.candidates)['per_query']
    anchor = read(args.admission)
    cached = {key(r): r for r in read(args.previous_inputs)['per_query']}
    if len(cases) != 191 or {key(r) for r in cases} != cached.keys():
        raise ValueError('Expected the same locked 191 topics')
    group = defaultdict(list)
    for row in cases:
        if row['contrast'] is not None:
            group[row['domain']].append(row)
    source = []
    reqs = {}
    large = {key(r): r for r in anchor['raw_scores']['large']}
    for domain, rows in group.items():
        originals = dict(zip(*load_examples(args.data_root/domain, 'train'), strict=True))
        for row in rows:
            question = key(row)
            first = large[question]
            if row['current'] != first['current'] or row['offers']['two_sided'] != first['sources'][2]:
                raise ValueError(f'Unexpected rank or offer on {question}')
            views = identify_fragments(originals[row['qid']])
            if len(views) != row['fragments']:
                raise ValueError(f'Literal requirements changed on {question}')
            selection = row['contrast']
            reqs[question] = {view: views[selection[view]] for view in VIEWS}
            source.append({k: row[k] for k in ('domain', 'qid', 'panel', 'current')} |
                          {'docs': row['current'] + row['offers']['two_sided'],
                           'sources': first['sources'], 'prior_score': first['prior_score']})
    args.official = False
    prepared = prepare(args, source)['per_query']
    output = []
    for row in prepared:
        old = cached[key(row)]
        selected = dict(zip(row['docs'], row['excerpts'], strict=True))
        for doc, text in zip(old['docs'], old['excerpts'], strict=True):
            if doc in selected and selected[doc] != text:
                raise ValueError(f'Reconstructed excerpt differs for {key(row)}/{doc}')
        output.extend(row | {'view': name, 'query': text} for name, text in reqs[key(row)].items())
    return {'per_query': output, 'queries': len(prepared)}


def changes(candidate, incumbent, spare_max):
    return max(candidate, spare_max) - max(incumbent, spare_max)


def choose(current, offers, utilities):
    item = max(offers, key=utilities.__getitem__)
    return current[:9] + [item] if utilities[item] > 0 else current


def rank_rows(cases, whole, scores):
    rows = []
    by_view = {view: {key(r): r for r in scores['per_query'] if r['view'] == view} for view in VIEWS}
    active = {key(r) for r in cases if r['contrast'] is not None}
    if any(part.keys() != active for part in by_view.values()) or len(scores['per_query']) != 2*len(active):
        raise ValueError('Expected aligned scores for each active requirement')
    if scores['model'] != whole['models'][0]:
        raise ValueError('All text views must use the same checkpoint')
    previous = {key(r): r for r in whole['raw_scores']['large']}
    for case in cases:
        keyid = key(case)
        current, offers = case['current'], case['offers']['two_sided']
        first = previous[keyid]
        if len(current) != 10 or len(set(current + offers)) != 14 or first['sources'][2] != offers:
            raise ValueError(f'Expected fixed nonoverlapping lists for {keyid}')
        kept, incumbent = current[:9], current[9]
        old = dict(zip(first['docs'], first['new_scores'], strict=True))
        utilities = {}
        head_support = {}
        if keyid in active:
            utilities['whole'] = {d: old[d]-old[incumbent] for d in offers}
            raw = []
            for view in VIEWS:
                row = by_view[view][keyid]
                if row['docs'] != current+offers or row['current'] != current:
                    raise ValueError(f'Inputs changed on {keyid}')
                vector = dict(zip(row['docs'], row['new_scores'], strict=True))
                raw.append(vector)
                head_support[view] = max(vector[d] for d in kept)
            g, c = raw
            utilities['gap_only'] = {d: g[d]-g[incumbent] for d in offers}
            utilities['pair'] = {d: g[d]-g[incumbent]-max(0., c[incumbent]-c[d]) for d in offers}
            utilities['list_conditioned'] = {
                d: changes(g[d], g[incumbent], head_support['gap'])
                - max(0., -changes(c[d], c[incumbent], head_support['covered']))
                for d in offers
            }
        orders = {'current': current} | {name: choose(current, offers, utilities[name])
                                         if name in utilities else current for name in ARMS}
        rows.append({k: case[k] for k in ('domain', 'qid', 'panel')} |
                    {'prior_score': first['prior_score'], 'rankings': orders,
                     'head_support': head_support, 'utilities': utilities})
    return rows


def evaluate(args):
    cases = read(args.candidates)['per_query']
    whole, raw = read(args.admission), read(args.focused)
    rows = rank_rows(cases, whole, raw)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    rows = measure(rows, labels(rows, args.data_root))
    step_labels = {d: read_steps(args.data_root/d/'qrels_steps_train.txt') for d in {r['domain'] for r in rows}}
    counts = {a: Counter(changed=0, gained=0, lost=0) for a in ARMS}
    for row in rows:
        steps = step_labels[row['domain']].get(row['qid'], {})
        rankings = row['rankings']
        covered = {a: {step for step, docs in steps.items() if not docs.isdisjoint(order)}
                   for a, order in rankings.items()}
        for arm in ARMS:
            counts[arm].update(changed=rankings[arm] != rankings['current'],
                               gained=len(covered[arm]-covered['current']),
                               lost=len(covered['current']-covered[arm]))
    values = {a: macro(rows, a) for a in ('current', *ARMS)}
    contrasts = {a: effect(rows, 'list_conditioned', a) for a in ('current', 'pair', 'gap_only', 'whole')}
    gate = all(min(contrasts[a]['point'], contrasts[a]['minimum_leave_one_domain_out']) > 0
               for a in ('current', 'pair'))
    report = {'queries': len(rows), 'active': sum(bool(r['head_support']) for r in rows),
              'model': raw['model'], 'scores': values, 'changes': counts, 'contrasts': contrasts,
              'rank_fingerprint': digest, 'eligible': gate, 'per_query': rows,
              'focused_scores': raw['per_query']}
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_query','focused_scores')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    prep = stages.add_parser('prepare')
    prep.add_argument('--previous-inputs', type=Path, required=True)
    score = stages.add_parser('score')
    score.add_argument('--focused', type=Path, required=True)
    for stage in (prep, score):
        for arg in ('candidates', 'admission', 'data-root', 'output'):
            stage.add_argument(f'--{arg}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    write_gz(args.output, build(args) if args.stage == 'prepare' else evaluate(args))


if __name__ == '__main__':
    main()
