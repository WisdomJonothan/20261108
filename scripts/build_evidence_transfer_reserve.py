"""Reserve official training topics outside the archived 969 and build their pools."""
import argparse
from collections import Counter
import json
from pathlib import Path

from compare_anchor_reference import load
from extend_evidence_adapter_pools import build
from lib_reteco import load_examples, read_trec_run


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, separators=(',', ':'))


def prepare(data, archive, runs):
    prior = {(r['domain'], r['qid']) for r in load(archive)['queries']}
    if len(prior) != 969:
        raise ValueError('Expected the archived 969 official training topics')
    rows = []
    for examples in sorted(data.glob('*/examples_train.jsonl')):
        domain = examples.parent.name
        ids, texts = load_examples(examples.parent, 'train')
        unused = [(str(qid), query) for qid, query in zip(ids, texts, strict=True)
                  if (domain, str(qid)) not in prior]
        dense = read_trec_run(runs / f'dense_mxbai_{domain}_train.trec')
        fused = read_trec_run(runs / f'trainfuse_fall_{domain}_train.trec')
        for qid, query in unused:
            retrieved = [doc for _, _, doc in dense[qid][:110]]
            head = [doc for _, _, doc in fused[qid][:10]]
            if len(set(retrieved)) != 110 or len(set(head)) != 10 or not set(head) <= set(retrieved[:100]):
                raise ValueError(f'Unexpected reserve input runs for {domain}/{qid}')
            mid = [doc for doc in retrieved[:100] if doc not in head][:10]
            rows.append({'domain': domain, 'qid': qid, 'query': query,
                         'split': 'train', 'cohort': 'fall', 'actual_reference': False,
                         'docs': head + mid + retrieved[100:]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'archived-manifest', 'runs-root', 'bge-root', 'repo-root', 'output-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(args.data_root, args.archived_manifest, args.runs_root)
    counts = Counter(row['domain'] for row in rows)
    if len(rows) != 242 or len(counts) != 13:
        raise ValueError('Expected the fixed 242-question complement in 13 domains')
    args.output_dir.mkdir()
    first = args.output_dir / 'first30.json'
    write(first, {'queries': rows})
    write(args.output_dir / 'reserve-ids.json',
          {'queries': [[r['domain'], r['qid']] for r in rows], 'domains': dict(counts)})
    wide, _ = build(args.repo_root, [first], args.runs_root, args.bge_root, args.bge_root)
    write(args.output_dir / 'wide60.json', wide)
    print('Reserved', len(rows), 'questions across', len(counts), 'domains', flush=True)


if __name__ == '__main__':
    main()
