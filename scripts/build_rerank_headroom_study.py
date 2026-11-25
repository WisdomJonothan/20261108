"""Build label-free candidate pools and audit ranking and retrieval headroom."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_evidence_shortlist import query_text


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def query_key(row):
    return row['domain'], row['qid']


def build(root, runs, data, extra_per_domain):
    results = root / 'results' / 'analysis'
    protected = load(results / 'accepted_weight_full_dev_audit_20261125.json')
    new = load(results / 'directional_passage_containment_20261125.json')['per_query']
    scoped = {
        part: ({query_key(r): r['rankings']['1'] for r in protected[part]['per_query']} |
               {query_key(r): r['rankings']['double90'] for r in new if r['sample'] == part})
        for part in ('dev', 'oof')
    }
    if len(scoped['dev']) != 519 or len(scoped['oof']) != 65:
        raise ValueError('Unexpected reference population sizes')
    oof_split = {}
    for fold in ('f0', 'f1', 'f2'):
        for r in load(root / 'results' / 'replacement_v2' / f'candidates_train_{fold}.json')['queries']:
            key = query_key(r)
            if key in oof_split:
                raise ValueError(f'Query occurs in different folds: {key}')
            oof_split[key] = fold, r['frozen_top10']
    routing = index(load(results / 'position_oracle_oof_20261124.json'))
    raw_scores = {}
    for name in ('anchor_reranker_qwen8b_sample26', 'anchor_reranker_qwen8b_replication39',
                 'anchor_reranker_qwen8b_dev100'):
        archived = index(load(results / f'{name}_20261125.json'))
        if not raw_scores.keys().isdisjoint(archived):
            raise ValueError('Query occurs in multiple 8B cohorts')
        raw_scores.update(archived)
    fused = {}
    for name in ('oof', 'dev'):
        fused.update(index(load(results / f'anchor_edit_scores_{name}100_20261125.json')))

    extra = defaultdict(list)
    for key in routing:
        if key not in scoped['oof']:
            extra[key[0]].append(key)
    chosen = set(scoped['oof'])
    for keys in extra.values():
        order = sorted(keys, key=lambda key: hashlib.sha256(('rerank-heldout|' + '|'.join(key)).encode()).digest())
        chosen.update(order[:extra_per_domain])

    labels = {}
    output, audit = [], []
    for split, scope, selected in (('train', 'oof', chosen), ('dev', 'dev', scoped['dev'])):
        queries = query_text(root, split)
        domains = sorted({d for d, _ in selected})
        for domain in domains:
            ranked = read_trec_run(runs / f'dense_mxbai_{domain}_{split}.trec')
            labels[domain] = read_qrels(data / domain / f'qrels_{split}.txt')
            for key in sorted(k for k in selected if k[0] == domain):
                _, qid = key
                fold = 'dev' if split == 'dev' else oof_split[key][0]
                dense = [doc for _, _, doc in ranked[qid][:300]]
                if len(dense) != 300 or len(set(dense)) != 300:
                    raise ValueError(f'Expected 300 distinct dense documents for {key}')
                if key in scoped[scope]:
                    start = scoped[scope][key]
                    old = fused[key]['frozen_scores']
                    judge = raw_scores[key]['scores']
                    if set(dense[:100]) != judge.keys() or {doc for doc, _ in old} != judge.keys():
                        raise ValueError(f'Cached first-stage candidates differ for {key}')
                    first = .7 * z([v for _, v in old])
                    first += .3 * z([judge[d] for d, _ in old])
                    in_pool = [old[i][0] for i in sorted(range(len(old)), key=lambda i: -first[i])]
                else:
                    start = oof_split[key][1].copy()
                    change = routing[key]['policies']['external6']['chosen']
                    if change is not None:
                        slot = change['slot'] - 1
                        if start[slot] != change['incumbent']:
                            raise ValueError(f'Changed incumbent differs for {key}')
                        start[slot] = change['challenger']
                    in_pool = dense[:100]
                if len(set(start)) != 10 or not set(start) <= set(dense[:100]):
                    raise ValueError(f'Invalid reference ranking for {key}')
                near = [doc for doc in in_pool if doc not in start][:10]
                deep = dense[100:110]
                docs = start + near + deep
                output.append({'domain': domain, 'qid': qid, 'split': split, 'cohort': fold,
                               'actual_reference': key in scoped[scope], 'docs': docs,
                               'query': queries[key]})
                gold = labels[domain][qid]
                ideal = lambda pool: ndcg10(sorted(set(pool) & gold), gold)
                head_fixed = [start[0]] + sorted(set(start[1:]) & gold)
                audit.append({'domain': domain, 'qid': qid, 'split': split, 'scores': {
                    'reference': ndcg10(start, gold), 'reorder_same_ten': ideal(start),
                    'reorder_same_ten_head_fixed': ndcg10(head_fixed, gold),
                    'within_twenty': ideal(docs[:20]), 'expanded_thirty': ideal(docs),
                    'dense100': ideal(dense[:100]), 'dense300': ideal(dense),
                }})
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--extra-per-domain', type=int, default=10)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if args.extra_per_domain < 0 or args.manifest.exists() or args.audit.exists():
        parser.error('Use a nonnegative sample cap and fresh output paths')
    manifest, audit = build(args.root, args.runs_root, args.data_root, args.extra_per_domain)
    summaries = {}
    for split in ('train', 'dev'):
        rows = [r for r in audit if r['split'] == split]
        summaries[split] = {}
        for name in rows[0]['scores']:
            values = [(r['domain'], r['scores'][name]) for r in rows]
            gain = [(r['domain'], v - r['scores']['reference']) for (_, v), r in zip(values, rows, strict=True)]
            summaries[split][name] = {
                'score': paired_delta(values)['point'], 'change': paired_delta(gain),
                'zero_queries': sum(v == 0 for _, v in values),
            }
        print(split, len(rows), {k: v['score'] for k, v in summaries[split].items()}, flush=True)
    if abs(summaries['dev']['reference']['score'] - .39269165354596675) > 1e-12:
        raise ValueError('Official baseline differs from the previously audited system')
    for path, record in ((args.manifest, {'queries': manifest}),
                         (args.audit, {'summary': summaries, 'per_query': audit})):
        with path.open('x', encoding='utf-8') as file:
            json.dump(record, file, separators=(',', ':'))
            file.write('\n')


if __name__ == '__main__':
    main()
