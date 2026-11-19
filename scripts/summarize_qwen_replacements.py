import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from lib_reteco import read_trec_run

source = Path('results/pairwise_replacement_qwen4b_holdout214.json')
raw = json.loads(source.read_text())
groups = defaultdict(list)
for row in raw['per_pair']:
    groups[row['domain'], row['qid']].append(row)
split = json.loads(Path('results/fusion100_split.json').read_text())['eval']
assert set(groups) == {(domain, qid) for domain, ids in split.items() for qid in ids}
assert len(groups) == 214 and all(len(rows) == 3 for rows in groups.values())
metrics = defaultdict(lambda: defaultdict(list))
counts = {mode: Counter() for mode in ('direct', 'contextual')}
results = []
for domain in sorted(split):
    data = Path('data/track1_tempo') / domain
    run = read_trec_run(Path('runs') / f'dense_mxbai_{domain}_train_head20.trec')
    gold = read_qrels(data / 'qrels_train.txt')
    steps = read_steps(data / 'qrels_steps_train.txt')
    for qid in sorted(split[domain]):
        top = [doc for _, _, doc in run[qid][:10]]
        candidates = groups[domain, qid]
        row = {'domain': domain, 'qid': qid, 'baseline': {'ndcg': ndcg10(top, gold[qid])},
               'oracle_pair_delta': max(0, *(r['swap_delta_ndcg'] for r in candidates))}
        row['baseline']['coverage'] = (sum(bool(set(top) & ids) for ids in steps[qid].values()) /
                                       len(steps[qid]) if steps[qid] else None)
        for mode in counts:
            best_margin, selected = 0.0, None
            for pair in candidates:
                votes = [v['probabilities'] for v in pair['predictions'][mode]]
                means = {label: sum(math.log(max(v[label], 1e-15)) for v in votes)/2 for label in votes[0]}
                margin = means['challenger'] - max(means['incumbent'], means['abstain'])
                if margin > best_margin:
                    best_margin, selected = margin, pair
            value = {**row['baseline'], 'swap': None}
            if selected is not None:
                delta = selected['swap_delta_ndcg']
                value['ndcg'] += delta
                if value['coverage'] is not None:
                    value['coverage'] += selected['swap_delta_steps'] / selected['steps']
                value['swap'] = {k: selected[k] for k in ('slot', 'incumbent_id', 'challenger_id')}
                counts[mode]['helpful' if delta > 1e-12 else 'harmful' if delta < -1e-12 else 'neutral'] += 1
            else:
                counts[mode]['kept'] += 1
            row[mode] = value
        for mode in ('baseline', *counts):
            metrics[mode][domain].append(row[mode])
        results.append(row)
summary = {}
for mode, domains in metrics.items():
    per_domain = {d: {metric: float(np.mean(values)) if values else None for metric in ('ndcg','coverage')
                     if (values := [r[metric] for r in rows if r[metric] is not None]) or metric == 'coverage'}
                  for d, rows in domains.items()}
    summary[mode] = {'per_domain': per_domain, 'counts': dict(counts.get(mode, {})),
                     **{f'macro_{m}': float(np.mean([r[m] for r in per_domain.values() if r[m] is not None]))
                        for m in ('ndcg','coverage')}}
output = {'scope': 'Exploratory internal holdout. At most one swap, highest positive symmetric mean log-probability margin across fixed pairs. No labels used to select swaps. Same two prompts per pair and same candidate pairs. Contextual prompts include additional input tokens.',
          'queries': len(results), 'domains': len(split), 'pairs': raw['pairs'], 'prompts': raw['prompts'],
          'scoring_seconds': raw['scoring_seconds'], 'order_agreement': raw['metrics'],
          'queries_with_helpful_pair': sum(r['oracle_pair_delta'] > 1e-12 for r in results),
          'summary': summary, 'per_query': results}
Path('results/pairwise_replacement_qwen4b_holdout214_summary.json').write_text(json.dumps(output, indent=2))
print({k: {m: v for m, v in s.items() if m != 'per_domain'} for k, s in summary.items()})
print('queries_with_helpful_pair', output['queries_with_helpful_pair'])

rng = np.random.default_rng(20261119)
comparisons = {}
for mode, reference in [('direct', 'baseline'), ('contextual', 'baseline'), ('contextual', 'direct')]:
    changes = np.array([summary[mode]['per_domain'][domain]['ndcg'] -
                        summary[reference]['per_domain'][domain]['ndcg'] for domain in sorted(split)])
    draws = changes[rng.integers(len(changes), size=(10000, len(changes)))].mean(1)
    comparisons[f'{mode}_vs_{reference}'] = {'delta': float(changes.mean()),
                                             'domain_bootstrap_95': np.quantile(draws, [.025, .975]).tolist()}
output['paired_exploratory_comparisons'] = comparisons
output['oracle_macro_ndcg'] = float(np.mean([np.mean([r['baseline']['ndcg'] + r['oracle_pair_delta']
                                                       for r in results if r['domain'] == domain])
                                            for domain in sorted(split)]))
Path('results/pairwise_replacement_qwen4b_holdout214_summary.json').write_text(json.dumps(output, indent=2))
print(comparisons)
print('matched_pair_oracle_macro_ndcg', output['oracle_macro_ndcg'])
