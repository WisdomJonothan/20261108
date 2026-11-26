"""Create one wider candidate pool while retaining the fixed 30-document study."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import z
from replay_anchor_fallback import index


def picks(documents, excluded, count):
    chosen = []
    for doc in documents:
        if doc not in excluded:
            chosen.append(doc)
            excluded.add(doc)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise ValueError('Candidate source has too few unique documents')
    return chosen


def build(root, manifests, run_root, bge_train, bge_dev, data):
    all_rows = index({'per_query': [r for path in manifests for r in load(path)['queries']]})
    analysis = root / 'results/analysis'
    cached = {}
    for tag in ('sample26', 'replication39', 'dev100'):
        raw = index(load(analysis / f'anchor_reranker_qwen8b_{tag}_20261125.json'))
        if not raw.keys().isdisjoint(cached):
            raise ValueError('Duplicate original 8B results')
        cached.update(raw)
    by_domain = defaultdict(list)
    for r in all_rows.values():
        by_domain[r['split'], r['domain']].append(r)
    manifest = []
    effects = []
    for (split, domain), rows in sorted(by_domain.items()):
        dense = read_trec_run(run_root / f'dense_mxbai_{domain}_{split}.trec')
        alternative = bge_dev if split == 'dev' else bge_train
        bge = read_trec_run(alternative / f'dense_bge_{domain}_{split}.trec')
        frozen = {}
        for fold in sorted({r['cohort'] for r in rows}):
            tag = 'sub_sft_e2bc' if fold == 'dev' else f'trainfuse_{fold}'
            frozen[fold] = read_trec_run(run_root / f'{tag}_{domain}_{split}.trec')
        qrels = read_qrels(data / domain / f'qrels_{split}.txt')
        for r in rows:
            qid = r['qid']
            seen = set(r['docs'])
            source = [d for _, _, d in dense[qid][:300]]
            scored = frozen[r['cohort']][qid][:100]
            if len(seen) != 30 or len(source) != 300 or set(source[:100]) != {d for _, _, d in scored}:
                raise ValueError(f'Unexpected source pool for {domain}/{qid}')
            fused = z([s for _, s, _ in scored])
            if (domain, qid) in cached:
                judge = cached[domain, qid]['scores']
                fused = .7*fused + .3*z([judge[d] for _, _, d in scored])
            ordered = [scored[i][2] for i in sorted(range(100), key=lambda i: -fused[i])]
            inside = picks(ordered, seen, 15)
            tail = picks(source[100:], seen, 7)
            outside100 = set(source[:100]) | seen
            lexical = picks((d for _, _, d in bge[qid][:300]), outside100, 8)
            expanded = [*r['docs'], *inside, *tail, *lexical]
            if len(set(expanded)) != 60:
                raise ValueError(f'Expanded pool has duplicates for {domain}/{qid}')
            manifest.append(r | {'docs': expanded})
            gold = qrels[qid]
            effects.append({'domain': domain, 'qid': qid, 'split': split,
                            'oracle': {str(w): ndcg10(sorted(set(expanded[:w]) & gold), gold)
                                       for w in (30, 45, 60)}})
    return {'queries': manifest}, {'per_query': effects, 'summary': {
        s: {str(w): {'score': macro((r['domain'], r['oracle'][str(w)]) for r in effects if r['split'] == s),
                     'zero_queries': sum(r['oracle'][str(w)] == 0 for r in effects if r['split'] == s)}
            for w in (30, 45, 60)}
        for s in sorted({r['split'] for r in effects})
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('runs-root', 'bge-train-root', 'bge-dev-root', 'data-root', 'output', 'audit'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        parser.error('Use fresh output paths')
    manifest, summary = build(args.root, args.manifests, args.runs_root, args.bge_train_root,
                              args.bge_dev_root, args.data_root)
    if len(manifest['queries']) != 705:
        raise ValueError('Expected the 186 held-out and 519 official queries')
    print(summary['summary'], flush=True)
    for path, payload in ((args.output, manifest), (args.audit, summary)):
        with path.open('x', encoding='utf-8') as file:
            json.dump(payload, file, separators=(',', ':'))
            file.write('\n')


if __name__ == '__main__':
    main()
