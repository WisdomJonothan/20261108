"""Measure top-ten position exchanges and external replacements at equal judge budget."""
import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import benchmark_pairwise_replacement_v2 as judge
from lib_reteco import read_trec_run


BUDGET = 6
DISCOUNTS = [1 / math.log2(rank + 2) for rank in range(10)]
DEV_CE = 'fusion100_full_ce_s17_20261115'
POLICIES = ('external6', 'internal6', 'mixed3x3')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def select_internal(top, scores):
    values = np.array(list(scores.values()))
    scale = max(float(values.std()), 1e-8)
    return sorted(itertools.combinations(range(10), 2), key=lambda ij: (
        -(scores[top[ij[1]]] - scores[top[ij[0]]]) / scale
        * (DISCOUNTS[ij[0]] - DISCOUNTS[ij[1]]), ij))[:BUDGET]


def build_rows(root, fold, load_text):
    split = 'dev' if fold == 'dev' else 'train'
    artifacts = root / 'results/replacement_v2'
    suffix = 'dev' if fold == 'dev' else f'train_{fold}'
    manifest = read_json(artifacts / f'candidates_{suffix}.json')
    cached = read_json(artifacts / f'expanded3_fixed_{suffix}.json')
    historical = {(r['domain'], r['qid']): r for r in cached['budgets']['18']['per_query']}
    by_domain = defaultdict(list)
    for row in manifest['queries']:
        by_domain[row['domain']].append(row)
    output = []
    for domain, queries in sorted(by_domain.items()):
        data = root / 'data/track1_tempo' / domain
        tag = DEV_CE if fold == 'dev' else f'{fold}_ce'
        ce = read_trec_run(root / 'runs' / f'{tag}_{domain}_{split}_ce.trec')
        needed = {doc for row in queries for doc in row['frozen_top10']}
        corpus = {}
        if load_text:
            with (data / 'documents.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    if record['id'] in needed:
                        corpus[record['id']] = record['content']
                        if len(corpus) == len(needed):
                            break
            assert set(corpus) == needed
        gold = judge.read_qrels(data / f'qrels_{split}.txt')
        for row in queries:
            qid, top = row['qid'], row['frozen_top10']
            assert len(set(top)) == 10
            scores = {doc: score for _, score, doc in ce[qid]}
            internal = select_internal(top, scores)
            baseline = judge.ndcg10(top, gold[qid])
            reference = historical[domain, qid]
            assert abs(reference['baseline'] - baseline) < 1e-10
            external = sorted(reference['actions'], key=lambda a: (a['slot'], -a['prior']))[:BUDGET]
            assert len(external) == BUDGET
            for action in external:
                assert action['incumbent'] == top[action['slot'] - 1]
                assert action['challenger'] not in top
            query = judge.strip_html(row['query'])
            terms = judge.terms(query)
            docs_to_score = {top[i] for pair in internal for i in pair}
            excerpts = {doc: judge.passages(corpus[doc], terms)[0] for doc in docs_to_score} if load_text else {}
            pairs = []
            for i, j in internal:
                changed = top.copy()
                changed[i], changed[j] = changed[j], changed[i]
                pairs.append({'slot': i + 1, 'other_slot': j + 1, 'incumbent': top[i], 'challenger': top[j],
                              'incumbent_text': excerpts.get(top[i], ''),
                              'challenger_text': excerpts.get(top[j], ''), 'retained': [],
                              'delta': judge.ndcg10(changed, gold[qid]) - baseline})
            output.append({'domain': domain, 'qid': qid, 'query': query, 'baseline': baseline,
                           'pairs': pairs, 'external': external})
    assert len(output) == len(historical) == manifest['n_queries']
    return output


def internal_scores(pair):
    prediction = pair['pred']
    votes = [prediction[k] for k in ('fwd', 'rev')]
    return {'mp': sum(v['challenger'] for v in votes) / 2,
            'agree': all(max(v, key=v.get) == 'challenger' for v in votes)}


def decide(actions):
    return max((a for a in actions if a['agree'] and a['mp'] > .5),
               key=lambda a: a['mp'], default=None)


def audit(rows, scored):
    outputs = []
    for row in rows:
        internal = [{k: pair[k] for k in ('slot', 'other_slot', 'incumbent', 'challenger', 'delta')}
                    | (internal_scores(pair) if scored else {}) | {'type': 'internal'}
                    for pair in row['pairs']]
        external = [a | {'type': 'external'} for a in row['external']]
        arms = dict(external6=external, internal6=internal, mixed3x3=external[:3] + internal[:3])
        record = {k: row[k] for k in ('domain', 'qid', 'baseline')}
        record['actions'] = {'external': external, 'internal': internal}
        record['policies'] = {}
        for arm, actions in arms.items():
            chosen = decide(actions) if scored or arm == 'external6' else None
            record['policies'][arm] = {'max_delta': max(0., *(a['delta'] for a in actions)),
                                       'chosen': chosen, 'delta': chosen['delta'] if chosen else 0.}
        outputs.append(record)
    return outputs


def summarize(records, scored):
    baseline = judge.domain_macro(records, key='baseline')
    arms = {}
    for arm in POLICIES:
        values = [{k: row[k] for k in ('domain', 'qid')} | row['policies'][arm] for row in records]
        result = {'oracle': baseline + judge.domain_macro(values, key='max_delta'),
                  'reachable_queries': sum(v['max_delta'] > 1e-12 for v in values)}
        if scored or arm == 'external6':
            delta = judge.domain_macro(values, key='delta')
            signs = Counter('kept' if not v['chosen'] else 'helpful' if v['delta'] > 1e-12
                            else 'harmful' if v['delta'] < -1e-12 else 'neutral' for v in values)
            against_external = [{**v, 'delta': v['delta'] - row['policies']['external6']['delta']}
                                for v, row in zip(values, records)]
            result |= {'ndcg': baseline + delta, 'delta': delta, 'counts': signs,
                       'paired_bootstrap': judge.paired_bootstrap(values),
                       'delta_vs_external6': judge.domain_macro(against_external, key='delta'),
                       'paired_vs_external6': judge.paired_bootstrap(against_external)}
        arms[arm] = result
    return {'queries': len(records), 'domains': len({r['domain'] for r in records}),
            'baseline': baseline, 'arms': arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', type=Path)
    parser.add_argument('--folds', nargs='+', choices=('f0', 'f1', 'f2', 'dev'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if 'dev' in args.folds and len(args.folds) > 1:
        parser.error('Report official dev and train folds separately.')
    if args.output.exists():
        parser.error(f'Output exists: {args.output}')
    model, tokenizer, options = None, None, None
    if args.model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                    local_files_only=True).to('cuda').eval()
        tokens = [tokenizer.encode(s, add_special_tokens=False) for s in 'ABC']
        assert all(len(t) == 1 for t in tokens)
        options = [t[0] for t in tokens]
    records = []
    cost = {}
    for fold in args.folds:
        rows = build_rows(args.root, fold, load_text=model is not None)
        print(fold, 'queries', len(rows), flush=True)
        if model is not None:
            cost[fold] = judge.score_arm(model, tokenizer, rows, 'direct', 12, 'cuda', options)
        records.extend(audit(rows, scored=model is not None))
    assert len({(r['domain'], r['qid']) for r in records}) == len(records)
    summary = summarize(records, scored=model is not None)
    result = {'design': 'Six-pair budget. One exchange at most. In-list candidates prioritized by '
              'full-CE score advantage times the position discount difference. External actions reuse '
              'the head-first six-pair competition cache. Mixed uses the first three of each, with the '
              'same acceptance and maximum-confidence rule. Labels score outcomes only.',
              'folds': args.folds, 'judge_cost_internal_inference': cost, 'summary': summary,
              'per_query': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
