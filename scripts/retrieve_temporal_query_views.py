"""Compare same-width candidate pools from the original query and two evidence requests."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from extend_evidence_adapter_pools import picks
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


ARMS = ('existing60', 'full_query60', 'reciprocal60', 'interleaved60')


def read_views(paths, split):
    queries = {}
    models = None
    for path in paths:
        with np.load(path) as data:
            metadata = json.loads(data['metadata'].item())
            vectors = data['vectors']
        current = {k: metadata[k] for k in ('encoder', 'rewriter', 'instruction')}
        if metadata['split'] != split or (models is not None and models != current):
            raise ValueError('Query-view shards use different inputs')
        models = current
        for view, embedding in zip(metadata['views'], vectors, strict=True):
            key = view['domain'], view['qid']
            if key in queries or embedding.shape != (3, 1024) or not np.isfinite(embedding).all():
                raise ValueError(f'Invalid cached query views for {key}')
            queries[key] = view, embedding
    return queries, models


def retrieve(doc_ids, directory, rows, views):
    doc_vectors = np.load(directory / f'doc_emb_mxbai_{len(doc_ids)}.npy', mmap_mode='r')
    if doc_vectors.shape != (len(doc_ids), 1024):
        raise ValueError('Document embedding shape differs from the original mxbai index')
    matrix = np.concatenate([views[r['domain'], r['qid']][1] for r in rows])
    similarities = matrix @ doc_vectors.T
    top = np.argpartition(-similarities, kth=119, axis=1)[:, :120]
    scores = np.take_along_axis(similarities, top, axis=1)
    ranking = np.take_along_axis(top, np.argsort(-scores, axis=1, kind='stable'), axis=1)
    return np.asarray(doc_ids)[ranking].reshape(len(rows), 3, 120).tolist()


def pool_candidates(start, rankings):
    core = start[:30]
    full = picks(rankings[0], set(core), 30)
    reciprocal = defaultdict(float)
    for ranking in rankings:
        for i, key in enumerate(ranking):
            reciprocal[key] += 1/(61+i)
    fused = picks(sorted(reciprocal, key=lambda d: -reciprocal[d]), set(core), 30)
    interleaved = picks((d for triple in zip(*rankings, strict=True) for d in triple), set(core), 30)
    return dict(zip(ARMS, (start, core+full, core+fused, core+interleaved), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'embeddings', 'runs', 'output', 'audit'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        parser.error('Use fresh output paths')
    manifest = index({'per_query': load(args.manifest)['queries']})
    split, = {r['split'] for r in manifest.values()}
    views, models = read_views(args.vectors, split)
    if manifest.keys() != views.keys():
        raise ValueError('The viewpoint embeddings and widened candidate manifest differ')
    grouped = defaultdict(list)
    for (domain, qid), row in sorted(manifest.items()):
        if len(row['docs']) != 60 or len(set(row['docs'])) != 60:
            raise ValueError(f'Unexpected existing pool for {domain}/{qid}')
        grouped[domain].append(row)
    reports, expanded = [], {arm: [] for arm in ARMS}
    for domain, batch in sorted(grouped.items()):
        with (args.data_root / domain / 'documents.jsonl').open(encoding='utf-8') as file:
            doc_ids = [json.loads(line)['id'] for line in file if line.strip()]
        recovered = retrieve(doc_ids, args.embeddings, batch, views)
        control = read_trec_run(args.runs / f'dense_mxbai_{domain}_{split}.trec')
        labels = read_qrels(args.data_root / domain / f'qrels_{split}.txt')
        for row, three_ranks in zip(batch, recovered, strict=True):
            qid = row['qid']
            gold = labels[qid]
            baseline = {doc for _, _, doc in control[qid][:100]}
            agreement = len(baseline & set(three_ranks[0][:100])) / 100
            pools = pool_candidates(row['docs'], three_ranks)
            core = set(row['docs'][:30])
            effects = {}
            for arm, candidates in pools.items():
                relevant = set(candidates) & gold
                effects[arm] = {'oracle': ndcg10(sorted(relevant), gold), 'relevant': len(relevant),
                                'extra_relevant': len(relevant-core)}
                expanded[arm].append(row | {'docs': candidates})
            reports.append({'domain': domain, 'qid': qid, 'reproduced_top100': agreement,
                            'views': views[domain, qid][0], 'effects': effects})
        print(domain, len(reports), '/', len(manifest), 'queries retrieved', flush=True)
    overlap = sum(r['reproduced_top100'] for r in reports) / len(reports)
    if overlap < .9:
        raise ValueError(f'Original mxbai top-100 reproduction fell below 90% ({overlap:.3f})')
    summary = {arm: {'oracle': macro((r['domain'], r['effects'][arm]['oracle']) for r in reports),
                     'change': paired_delta([(r['domain'], r['effects'][arm]['oracle']
                                              - r['effects']['existing60']['oracle']) for r in reports]),
                     'without_relevant': sum(r['effects'][arm]['relevant'] == 0 for r in reports),
                     'new_relevant': sum(r['effects'][arm]['extra_relevant'] for r in reports)} for arm in ARMS}
    report = {'split': split, 'models': models, 'reproduced_top100': overlap, 'summary': summary, 'per_query': reports}
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(expanded, file, separators=(',', ':'))
    with args.audit.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
    print({arm: (v['oracle'], v['without_relevant'], v['new_relevant']) for arm, v in summary.items()}, flush=True)


if __name__ == '__main__':
    main()
