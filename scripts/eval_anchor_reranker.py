"""Rank the same dense top 100 by judgments against a shared in-list reference."""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmark_pairwise_replacement_v2 as judge
from lib_reteco import read_trec_run
from replay_position_discount import bootstrap


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build(root, queries_per_domain, exclude, pool_depth):
    directory = root / 'results/replacement_v2'
    source = []
    for fold in ('f0', 'f1', 'f2'):
        for row in load(directory / f'candidates_train_{fold}.json')['queries']:
            source.append(row | {'fold': fold})
    rows = []
    for domain in judge.DOMAINS:
        candidates = sorted((r for r in source if r['domain'] == domain
                             and (domain, r['qid']) not in exclude), key=lambda r: r['qid'])
        selected = random.Random(f'anchor-20261124-{domain}').sample(
            candidates, min(queries_per_domain, len(candidates)))
        assert selected, domain
        base = {f: read_trec_run(root / 'runs' / f'trainfuse_{f}_{domain}_train.trec')
                for f in {r['fold'] for r in selected}}
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_train.trec')
        gold = judge.read_qrels(root / 'data/track1_tempo' / domain / 'qrels_train.txt')
        pools = {}
        for row in selected:
            full = [doc for _, _, doc in dense[row['qid']][:100]]
            assert len(set(full)) == 100 and set(row['frozen_top10']) <= set(full)
            members = set(list(dict.fromkeys(row['frozen_top10'] + full))[:pool_depth])
            pools[row['qid']] = [doc for doc in full if doc in members]
        need = {doc for pool in pools.values() for doc in pool}
        corpus = {}
        with (root / 'data/track1_tempo' / domain / 'documents.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                doc = json.loads(line)
                if doc['id'] in need:
                    corpus[doc['id']] = doc['content']
                    if len(corpus) == len(need):
                        break
        assert set(corpus) == need
        for row in selected:
            qid = row['qid']
            pool = pools[qid]
            pool_set = set(pool)
            top = row['frozen_top10']
            anchor = top[4]
            assert len(pool_set) == pool_depth and set(top) <= pool_set
            ranking = [doc for _, _, doc in base[row['fold']][qid] if doc in pool_set]
            assert len(ranking) == pool_depth and ranking[:10] == top
            base_scores = {doc: score for _, score, doc in base[row['fold']][qid] if doc in pool_set}
            query = judge.strip_html(row['query'])
            terms = judge.terms(query)
            passages = {doc: judge.passages(corpus[doc], terms)[0] for doc in pool}
            pairs = [{'doc': doc, 'incumbent_text': passages[anchor],
                      'challenger_text': passages[doc]} for doc in pool if doc != anchor]
            rows.append({'domain': domain, 'qid': qid, 'fold': row['fold'], 'query': query,
                         'top': top, 'pool': pool, 'anchor': anchor, 'gold': gold[qid],
                         'base_scores': base_scores, 'pairs': pairs})
        print('Prepared', domain, len(selected), flush=True)
    return rows


def z(values):
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def anchor_score(pair):
    # Average identity-aligned log odds to reduce first/second-position effects.
    votes = [pair['pred'][order] for order in ('fwd', 'rev')]
    return sum(math.log(max(v['challenger'], 1e-6))
               - math.log(max(v['incumbent'], 1e-6)) for v in votes) / 2


def evaluate(rows):
    details = []
    for r in rows:
        pool = r['pool']
        gold = r['gold']
        top = r['top']
        scores = {r['anchor']: 0.}
        scores.update({pair['doc']: anchor_score(pair) for pair in r['pairs']})
        values = z(np.array([scores[d] for d in pool]))
        old = z(np.array([r['base_scores'][d] for d in pool]))
        rankings = {'base': top, 'anchor_only': [pool[i] for i in np.argsort(-values, kind='stable')[:10]],
                    'base_plus_anchor30': [pool[i] for i in np.argsort(-(.7 * old + .3 * values), kind='stable')[:10]]}
        row = {k: r[k] for k in ('domain', 'qid', 'anchor', 'fold')}
        row['pool_positive'] = len(set(pool) & gold)
        row['anchor_relevant'] = r['anchor'] in gold
        row['scores'] = scores
        row['arms'] = {}
        for name, ranking in rankings.items():
            entering, leaving = set(ranking) - set(top), set(top) - set(ranking)
            row['arms'][name] = {'ndcg': judge.ndcg10(ranking, gold),
                                 'relevant_top10': len(set(ranking) & gold),
                                 'relevant_in': len(entering & gold),
                                 'relevant_out': len(leaving & gold), 'ranking': ranking}
        details.append(row)
    summary = {}
    for arm in ('base', 'anchor_only', 'base_plus_anchor30'):
        scores = [(r['domain'], r['arms'][arm]['ndcg']) for r in details]
        deltas = [(r['domain'], r['arms'][arm]['ndcg'] - r['arms']['base']['ndcg']) for r in details]
        summary[arm] = {'ndcg': bootstrap(scores)['point'], 'versus_base': bootstrap(deltas),
                        'relevant_top10': sum(r['arms'][arm]['relevant_top10'] for r in details),
                        'relevant_in': sum(r['arms'][arm]['relevant_in'] for r in details),
                        'relevant_out': sum(r['arms'][arm]['relevant_out'] for r in details),
                        'recovered_zeroes': sum(r['arms']['base']['ndcg'] <= 1e-12
                                                and r['arms'][arm]['ndcg'] > 1e-12 for r in details),
                        'new_zeroes': sum(r['arms']['base']['ndcg'] > 1e-12
                                          and r['arms'][arm]['ndcg'] <= 1e-12 for r in details)}
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--queries-per-domain', type=int, default=2)
    parser.add_argument('--pool-depth', type=int, choices=(20, 100), default=100)
    parser.add_argument('--exclude-report', type=Path,
                        help='Exclude every query previously scored in this report.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.queries_per_domain < 1:
        parser.error('Use a new output and a positive query count.')
    exclude = {(r['domain'], r['qid']) for r in load(args.exclude_report)['per_query']} if args.exclude_report else set()
    rows = build(args.root, args.queries_per_domain, exclude, args.pool_depth)
    assert len({(r['domain'], r['qid']) for r in rows}) == len(rows)
    assert exclude.isdisjoint((r['domain'], r['qid']) for r in rows)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    token_ids = [tok.encode(s, add_special_tokens=False) for s in 'ABC']
    assert all(len(t) == 1 for t in token_ids)
    stats = judge.score_arm(model, tok, rows, 'direct', 12, 'cuda', [t[0] for t in token_ids])
    summary, details = evaluate(rows)
    report = {'protocol': 'Stratified sample from held-out training folds. Protect the original '
              'top ten and fill the specified pool depth from dense rank. '
              'Score each document against rank 5 of the frozen list using the same lexical '
              'window and both orders of the Qwen Direct prompt. Normalize log odds by query. '
              'Compare original list to standalone score and fixed 70:30 fusion. No official dev.',
              'queries': len(rows), 'domains': len({r['domain'] for r in rows}), 'pool_depth': args.pool_depth,
              'excluded_previous_queries': len(exclude), 'inference': stats,
              'summary': summary, 'per_query': details}
    assert stats['prompts'] == 2 * (args.pool_depth - 1) * len(rows)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
