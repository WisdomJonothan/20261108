"""Blend original-query scores with the strongest support across two extractive views."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_qwen_reranker_adapter import candidate_judges
from replay_anchor_edit_scope import z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('full', 'first', 'second'):
        parser.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    models = [candidate_judges(paths) for paths in (args.full, args.first, args.second)]
    if any(set(metadata) != {'pretrained'} for _, metadata, _ in models):
        raise ValueError('Use the same frozen checkpoint for all three inputs')
    identity = models[0][1]['pretrained']
    if any(info['pretrained'] != identity for _, info, _ in models):
        raise ValueError('Checkpoint identities differ')
    original, first, second = [raw['pretrained'] for raw, _, _ in models]
    if original.keys() != first.keys() or original.keys() != second.keys():
        raise ValueError('Input views have different query sets')
    rows = []
    for key, row in original.items():
        docs = row['scores']
        alternatives = [source[key] for source in (first, second)]
        if any(v['scores'].keys() != docs.keys() or (v['split'], v['cohort']) != (row['split'], row['cohort'])
               for v in alternatives):
            raise ValueError(f'Inputs differ in candidate IDs or cohort for {key}')
        scores = [z([v['scores'][d] for d in docs]) for v in (row, *alternatives)]
        best_view = z(np.maximum(scores[1], scores[2]))
        merged = .6*scores[0] + .4*best_view
        rows.append(row | {'scores': dict(zip(docs, merged.tolist(), strict=True))})
    report = {'model': identity['model'], 'model_revision': identity['revision'],
              'evidence_mode': 'dispersed', 'score_policy': 'full_plus_extractive_max',
              'inference': {label: model[2]['pretrained'] for label, model in
                            zip(('full', 'first', 'second'), models, strict=True)}, 'per_query': rows}
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
    print(len(rows), 'queries merged across three matched inputs', flush=True)


if __name__ == '__main__':
    main()
