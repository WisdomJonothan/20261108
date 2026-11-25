"""Test a fixed increase in judge weight on previously unused accepted-list queries."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_anchor_dev import macro
from audit_anchor_prompt_outliers import trim_words
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_anchor_reranker import document_scores, judge, write_checkpoint
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index

CAP = 20
WEIGHTS = (0.3, 0.6)


def select(reference, exclusions, split):
    excluded = set().union(*(index(load(path)).keys() for path in exclusions))
    groups = defaultdict(list)
    for key, row in index(load(reference)).items():
        action = row['policies']['external6']['chosen']
        if key not in excluded and action is not None:
            groups[key[0]].append((key[1], action))
    chosen = []
    for domain, rows in sorted(groups.items()):
        sample = (rows if split == 'dev' else random.Random(f'accepted-transfer-20261125-{domain}').sample(
            sorted(rows), min(CAP, len(rows))
        ))
        chosen.extend((domain, qid, action) for qid, action in sorted(sample))
    return chosen


def build(root, selected, split):
    candidates = {}
    sources = {'dev': 'candidates_dev.json'} if split == 'dev' else {
        fold: f'candidates_train_{fold}.json' for fold in ('f0', 'f1', 'f2')
    }
    for fold, filename in sources.items():
        for row in load(root / 'results/replacement_v2' / filename)['queries']:
            key = row['domain'], row['qid']
            if key in candidates:
                raise ValueError(f'Duplicate training fold record for {key}')
            candidates[key] = fold, row
    pools, dense = {}, {}
    result = []
    for domain, qid, action in selected:
        fold, row = candidates[domain, qid]
        key = domain, fold
        if key not in pools:
            filename = (f'sub_sft_e2bc_{domain}_dev.trec' if split == 'dev'
                        else f'trainfuse_{fold}_{domain}_train.trec')
            pools[key] = read_trec_run(root / 'runs' / filename)
        if split == 'dev' and domain not in dense:
            dense[domain] = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        values = pools[key][qid][:100]
        stamps = {d: score for _, score, d in values}
        initial = row['frozen_top10']
        if len(stamps) != 100 or [d for _, _, d in values[:10]] != initial:
            raise ValueError(f'Frozen scores disagree with reference for {domain}/{qid}')
        i = action['slot'] - 1
        if initial[i] != action['incumbent'] or action['challenger'] in initial:
            raise ValueError(f'Archived acceptance does not match list for {domain}/{qid}')
        accepted = initial[:i] + [action['challenger']] + initial[i + 1:]
        anchor = initial[4]
        full_z = dict(zip(stamps, z(np.array(list(stamps.values())))))
        order = [d for _, _, d in dense[domain][qid][:100]] if split == 'dev' else stamps
        if set(order) != set(stamps):
            raise ValueError(f'Dense and frozen top-100 pools differ for {domain}/{qid}')
        docs = [d for d in order if d in accepted]
        if len(docs) != 10:
            raise ValueError(f'Accepted list has a candidate outside the frozen pool for {domain}/{qid}')
        result.append({
            'domain': domain, 'qid': qid, 'fold': fold, 'anchor': anchor, 'accepted': accepted,
            'documents': docs, 'frozen_z': {d: float(full_z[d]) for d in docs},
            'query': judge.strip_html(row['query']),
        })
    return result


def score(rows, root, model_path, checkpoint, disable_thinking):
    checkpoint.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in 'ABC']
    if any(len(x) != 1 for x in labels) or len({x[0] for x in labels}) != 3:
        raise ValueError('Expected three distinct option tokens')
    options = {'enable_thinking': False} if disable_thinking else None
    totals = {'prompts': 0, 'prompt_tokens': 0, 'gpu_peak_mb': 0, 'cached_queries': 0}
    begin = time.perf_counter()
    for domain in sorted({r['domain'] for r in rows}):
        group = [r for r in rows if r['domain'] == domain]
        needed = set().union(*[set(r['documents'] + [r['anchor']]) for r in group])
        corpus = {}
        with (root / 'data/track1_tempo' / domain / 'documents.jsonl').open(encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line)
                if doc['id'] in needed:
                    corpus[doc['id']] = doc['content']
                    if len(corpus) == len(needed):
                        break
        if corpus.keys() != needed:
            raise ValueError(f'Incomplete corpus for {domain}')
        for r in group:
            docs = r['documents']
            terms = judge.terms(r['query'])
            windows = {d: trim_words(judge.passages(corpus[d], terms)[0], 2048, 256)
                       for d in dict.fromkeys(docs + [r['anchor']])}
            pairs = [{'doc': d, 'incumbent_text': windows[r['anchor']],
                      'challenger_text': windows[d]} for d in docs if d != r['anchor']]
            identity = hashlib.sha256(json.dumps(
                [str(model_path.resolve()), options, r['query'], r['anchor'], pairs],
                sort_keys=True,
            ).encode()).hexdigest()
            path = checkpoint / f'{domain}-{r["qid"]}.json'
            if path.exists():
                saved = load(path)
                if saved['identity'] != identity:
                    raise ValueError(f'Checkpoint fingerprint differs for {r["qid"]}')
                totals['cached_queries'] += 1
            else:
                run = {'query': r['query'], 'anchor': r['anchor'], 'pairs': pairs}
                stats = judge.score_arm(model, tokenizer, [run], 'direct', 12, 'cuda',
                                        [x[0] for x in labels], template_options=options)
                saved = {'identity': identity, 'inference': stats,
                         'scores': {d: v for d, v in document_scores(run).items() if d in docs}}
                write_checkpoint(path, saved)
            r['judge_scores'] = saved['scores']
            if r['judge_scores'].keys() != set(docs) or saved['inference']['prompts'] != 2 * len(pairs):
                raise ValueError(f'Invalid checkpoint scores for {r["qid"]}')
            for field in ('prompts', 'prompt_tokens'):
                totals[field] += saved['inference'][field]
            totals['gpu_peak_mb'] = max(totals['gpu_peak_mb'], saved['inference']['gpu_peak_mb'])
        print('Finished', domain, len(group), 'total prompts', totals['prompts'], flush=True)
    return totals | {'wall_seconds': time.perf_counter() - begin}


def evaluate(rows, data, split):
    qrels = {d: read_qrels(data / d / f'qrels_{split}.txt') for d in {r['domain'] for r in rows}}
    records = []
    for r in rows:
        docs = r['documents']
        a = np.array([r['frozen_z'][d] for d in docs])
        b = z(np.array([r['judge_scores'][d] for d in docs]))
        rankings = {'start': r['accepted']}
        for alpha in WEIGHTS:
            fused = (1 - alpha) * a + alpha * b
            rankings[str(alpha)] = [docs[i] for i in np.argsort(-fused, kind='stable')]
        gold = qrels[r['domain']][r['qid']]
        records.append({k: v for k, v in r.items() if k != 'query'} | {
            'rankings': rankings, 'ndcg': {k: ndcg10(docs, gold) for k, docs in rankings.items()},
        })
    summary = {'macro': {k: macro((r['domain'], r['ndcg'][k]) for r in records)
                         for k in ('start', '0.3', '0.6')},
               'higher_weight': paired_delta([(r['domain'], r['ndcg']['0.6'] - r['ndcg']['0.3'])
                                               for r in records])}
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--exclude', type=Path, nargs='+', default=[])
    parser.add_argument('--split', choices=('train', 'dev'), default='train')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh result filename')
    if args.split == 'train' and not args.exclude:
        parser.error('Supply the earlier holdout reports to exclude.')
    if args.split == 'dev' and args.exclude:
        parser.error('Dev confirms every accepted list.')
    selected = select(args.reference, args.exclude, args.split)
    rows = build(args.root, selected, args.split)
    print('Selected', len(rows), 'queries in', len({r['domain'] for r in rows}), 'domains', flush=True)
    inference = score(rows, args.root, args.model, args.checkpoint, args.disable_thinking)
    records, summary = evaluate(rows, args.root / 'data/track1_tempo', args.split)
    report = {'model': args.model.name, 'split': args.split, 'disable_thinking': args.disable_thinking,
              'queries': len(records), 'domains': len({r['domain'] for r in records}),
              'summary': summary, 'inference': inference, 'per_query': records}
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
        file.write('\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
