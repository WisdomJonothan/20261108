"""Check whether lexical and semantic evidence windows make complementary decisions."""
import argparse
import json
from collections import Counter
from pathlib import Path

from replay_position_discount import bootstrap, eligible


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def choose(actions):
    return max(eligible(actions), key=lambda a: a['mp'], default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scored = load(args.input)
    reference = {(r['domain'], r['qid']): r['actions']['external']
                 for r in load(args.reference)['per_query']}
    policies = ('lexical', 'semantic', 'lexical_then_semantic', 'require_both')
    effects, counts = {p: [] for p in policies}, {p: Counter() for p in policies}
    details = []
    for row in scored['per_query']:
        domain, qid, base = row['domain'], row['qid'], row['baseline']
        lexical = reference[domain, qid]
        index = lambda a: (a['slot'], a['challenger'])
        replacement = {index(a): a for a in row['rescored_actions']}
        semantic = [replacement.get(index(a), a) for a in lexical]
        assert set(replacement) <= {index(a) for a in lexical}
        old, new = choose(lexical), choose(semantic)
        assert abs((old['delta'] if old else 0.) - row['lexical']['delta']) < 1e-12
        assert abs((new['delta'] if new else 0.) - row['semantic']['delta']) < 1e-12
        supported = {index(a): a for a in eligible(semantic)}
        paired = [a | {'mp': min(a['mp'], supported[index(a)]['mp'])}
                  for a in eligible(lexical) if index(a) in supported]
        both = choose(paired)
        decisions = dict(lexical=old, semantic=new, lexical_then_semantic=old or new, require_both=both)
        record = {'domain': domain, 'qid': qid, 'baseline': base, 'outcomes': {}}
        for policy, selected in decisions.items():
            delta = selected['delta'] if selected else 0.
            record['outcomes'][policy] = delta
            effects[policy].append((domain, delta - row['lexical']['delta']))
            counts[policy]['kept' if not selected else 'helpful' if delta > 1e-12
                           else 'harmful' if delta < -1e-12 else 'neutral'] += 1
        details.append(record)
    lexical_score = scored['summary']['scores']['lexical']
    summary = {}
    for p in policies:
        effect = bootstrap(effects[p])
        summary[p] = {'ndcg': lexical_score + effect['point'], 'versus_lexical': effect,
                      'counts': counts[p]}
    report = {'scope': 'Exploratory fold-only replay. Combined policies use both evidence '
              'judgments and extra calls when excerpts change. Fixed one-action limit.',
              'fold': scored['fold'], 'summary': summary, 'per_query': details}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
