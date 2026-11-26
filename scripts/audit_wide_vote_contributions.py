"""Account for saved top-ten nDCG changes by candidate origin and retained rank."""
import argparse
from collections import defaultdict
import json
from math import log2
from pathlib import Path
import tarfile

from official.scorer import load_qrels
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import discounted_changes


def member(archive, name):
    with tarfile.open(archive) as package:
        for item in package.getmembers():
            if item.name.removeprefix('./') == name:
                return json.load(package.extractfile(item))
    raise ValueError(f'Missing {name} in {archive}')


def decompose(rows, scopes, left, right, root):
    gold_cache = {}
    effects = defaultdict(list)
    changed_lists = 0
    for row in rows:
        domain, qid, split = (row[key] for key in ('domain', 'qid', 'split'))
        if (domain, split) not in gold_cache:
            gold_cache[domain, split] = load_qrels(root / domain / f'qrels_{split}.txt')
        gold = gold_cache[domain, split][qid]
        original = scopes[domain, qid]['original']
        first, tail = set(original[:30]), set(original[30:])
        before, after = (row['rankings'][name] for name in (left, right))
        changed_lists += before != after
        contribution = discounted_changes(before, after, gold)
        idcg = sum(1 / log2(i + 2) for i in range(min(10, len(gold))))
        entrants = {doc: 1 / (log2(rank + 2) * idcg) for rank, doc in enumerate(after)
                    if doc in gold and doc not in before}
        components = {
            'entrants_outside_original_60': sum(value for doc, value in entrants.items()
                                                if doc not in first and doc not in tail),
            'entrants_from_original_31_to_60': sum(value for doc, value in entrants.items()
                                                   if doc in tail),
            'entrants_from_original_first_30': sum(value for doc, value in entrants.items()
                                                    if doc in first),
            'departures': contribution['removed_documents'],
            'retained_rank_movement': contribution['retained_documents'],
        }
        delta = row['scores'][right] - row['scores'][left]
        if abs(sum(components.values()) - delta) > 1e-12:
            raise ValueError(f'Discounted effects differ from saved scores on {domain}/{qid}')
        for name, value in (components | {'total': delta}).items():
            effects[name].append((domain, value))
    return {
        'queries': len(rows), 'changed_lists': changed_lists,
        'effects': {name: paired_delta(values) for name, values in effects.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--ranking-member', required=True)
    parser.add_argument('--scopes', type=Path, required=True)
    parser.add_argument('--scope-member', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--baseline', default='current')
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = member(args.rankings, args.ranking_member)['per_query']
    scopes = {(r['domain'], r['qid']): r for r in member(args.scopes, args.scope_member)['queries']}
    if set(scopes) != {(r['domain'], r['qid']) for r in rows}:
        raise ValueError('The rankings and candidate scopes have different query IDs')
    result = {arm: decompose(rows, scopes, args.baseline, arm, args.data_root) for arm in args.arms}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
