"""Use the first three judgments to decide which comparisons get the last three slots."""
import argparse
import json
from collections import Counter
from pathlib import Path

from replay_position_discount import bootstrap, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding='utf-8'))
    rows, counts = [], Counter()
    for row in result['per_query']:
        assert len(row['control']) == len(row['treatment']) == 6
        assert row['control'][:3] == row['treatment'][:3]
        # A deployable gate: the first three comparisons are shared.
        source = 'control' if eligible(row['control'][:3]) else 'treatment'
        actions = row[source]
        selected = max(eligible(actions), key=lambda a: a['mp'], default=None)
        delta = selected['delta'] if selected else 0.
        assert abs(delta - row[f'{source}_delta']) < 1e-12
        counts['kept' if not selected else 'helpful' if delta > 1e-12
               else 'harmful' if delta < -1e-12 else 'neutral'] += 1
        rows.append({'domain': row['domain'], 'qid': row['qid'], 'source': source,
                     'delta_vs_original': delta - row['control_delta']})
    effect = bootstrap([(r['domain'], r['delta_vs_original']) for r in rows])
    out = {'scope': 'Exploratory replay. Judge the first three original actions. If any passes '
           'the fixed two-order acceptance gate, use the original remaining three; otherwise use '
           'the CE disagreement actions with original refill. Always six comparisons at most.',
           'queries': len(rows), 'uses_disagreement': sum(r['source'] == 'treatment' for r in rows),
           'ndcg': result['summary']['scores']['control']['ndcg'] + effect['point'],
           'paired_vs_original': effect, 'counts': counts, 'per_query': rows}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(out, stream, indent=1)
    print({k: v for k, v in out.items() if k != 'per_query'})


if __name__ == '__main__':
    main()
