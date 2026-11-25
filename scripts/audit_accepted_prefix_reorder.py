"""Attribute RETECO score changes and replay prefix edits with cached 8B scores."""
import argparse
import json
from pathlib import Path

from audit_anchor_dev import coverage, macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import discounted_changes, index

PREFIXES = (1, 3, 5, 8, 10)
TRANSITIONS = (('frozen', 'six_pair'), ('six_pair', 'fallback'), ('fallback', 'routed'))


def rankings(report, data, split):
    starts = report['starts']
    frozen = index(starts['frozen'])
    competition = index(starts['competition'])
    fallback = index(starts['abstention'])
    if frozen.keys() != competition.keys() or competition.keys() != fallback.keys():
        raise ValueError('Ranking reports must contain the same queries')
    domains = sorted({domain for domain, _ in competition})
    labels = {d: read_qrels(data / d / f'qrels_{split}.txt') for d in domains}
    steps = {d: read_steps(data / d / f'qrels_steps_{split}.txt') for d in domains} if split == 'dev' else {}
    rows = []
    for (domain, qid), r in competition.items():
        six = r['rankings']['start']
        same_ten = r['rankings']['reorder_same_ten']
        key = domain, qid
        abstained = fallback[key]['active']
        baseline = fallback[key]['rankings']['unrestricted']
        if set(same_ten) != set(six) or len(set(six)) != 10:
            raise ValueError(f'Membership changed in the same-ten arm for {key}')
        lists = {'frozen': frozen[key]['rankings']['start'], 'six_pair': six,
                 'fallback': baseline, 'routed': baseline if abstained else same_ten}
        for k in PREFIXES:
            prefix = set(six[:k])
            lists[f'prefix{k}'] = baseline if abstained else [doc for doc in same_ten if doc in prefix] + six[k:]
        gold = labels[domain][qid]
        scores = {name: ndcg10(documents, gold) for name, documents in lists.items()}
        if abs(scores['six_pair'] - r['scores']['start']) > 1e-12 or (
            abs(scores['fallback'] - fallback[key]['scores']['unrestricted']) > 1e-12
        ):
            raise ValueError(f'Baseline scores differ for {key}')
        if scores['prefix1'] != scores['fallback'] or scores['prefix10'] != scores['routed']:
            raise ValueError(f'Unexpected prefix limits for {key}')
        rows.append({'domain': domain, 'qid': qid, 'abstained': abstained, 'gold': gold,
                     'steps': steps.get(domain, {}).get(qid), 'scores': scores, 'rankings': lists})
    return rows


def prefixes(rows):
    return {
        str(k): {
            'ndcg': macro((r['domain'], r['scores'][f'prefix{k}']) for r in rows),
            'versus_fallback': paired_delta([
                (r['domain'], r['scores'][f'prefix{k}'] - r['scores']['fallback']) for r in rows
            ]),
        }
        for k in PREFIXES
    }


def attribute(rows):
    out = {}
    for before, after in TRANSITIONS:
        effects = []
        for row in rows:
            old, new = (row['rankings'][name] for name in (before, after))
            effect = discounted_changes(old, new, row['gold'])
            delta = row['scores'][after] - row['scores'][before]
            if abs(sum(effect['rank_delta']) - delta) > 1e-12:
                raise ValueError(f'Rank contributions do not sum for {row["qid"]}')
            effects.append((row['domain'], effect))
        out[f'{before}_to_{after}'] = {
            'paired_delta': paired_delta([(r['domain'], r['scores'][after] - r['scores'][before]) for r in rows]),
            **{name: macro((domain, effect[name]) for domain, effect in effects)
               for name in ('added_documents', 'removed_documents', 'retained_documents')},
            'by_rank': [macro((domain, effect['rank_delta'][i]) for domain, effect in effects)
                        for i in range(10)],
            'step_coverage': {
                name: macro((r['domain'], coverage(r['rankings'][name], r['steps'])) for r in rows if r['steps'])
                for name in (before, after)
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/track1_tempo'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = rankings(load(args.oof), args.data_root, 'train')
    dev = rankings(load(args.dev), args.data_root, 'dev')
    train_prefix, dev_prefix = (prefixes(rows) for rows in (train, dev))
    chosen = max(PREFIXES, key=lambda k: train_prefix[str(k)]['ndcg'])
    output = {
        'protocol': 'Reorder a prefix within the accepted six-pair list using the cached 8B order. '
                    'On abstentions, keep the depth-100 fallback. Choose prefix length on the 65 '
                    'held-out training queries and replay it on the 519 official dev queries.',
        'oof_queries': len(train), 'dev_queries': len(dev),
        'oof_prefixes': train_prefix, 'dev_prefixes': dev_prefix, 'oof_selected_prefix': chosen,
        'dev_attribution': attribute(dev),
        'dev_per_query': [
            {'domain': r['domain'], 'qid': r['qid'], 'ranking': r['rankings'][f'prefix{chosen}'],
             'ndcg': r['scores'][f'prefix{chosen}']} for r in dev
        ],
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, separators=(',', ':'))
        stream.write('\n')
    print('Selected prefix:', chosen)
    for k in PREFIXES:
        print(k, train_prefix[str(k)]['ndcg'], dev_prefix[str(k)]['ndcg'])
    for name, result in output['dev_attribution'].items():
        print(name, result['paired_delta']['point'], result['added_documents'],
              result['removed_documents'], result['retained_documents'],
              [sum(result['by_rank'][start:end]) for start, end in ((0,3),(3,6),(6,10))],
              result['step_coverage'])


if __name__ == '__main__':
    main()
