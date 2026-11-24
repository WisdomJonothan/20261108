"""Run the full-ranking Direct judge on the fixed temporal contrast pilot."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmark_pairwise_replacement_v2 as judge
from eval_anchor_reranker import anchor_score
from eval_temporal_query_contrast import SIDES, evaluate, prepare
from compare_anchor_reference import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--native-template', action='store_true')
    parser.add_argument('--label', default='direct_qwen8b')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output.')
    examples = prepare(load(args.manifest)['cases'], args.data_root)
    rows = [{'query': queries[side], 'pairs': [{
        'doc': case['evidence']['late']['doc'], 'incumbent_text': docs['early'],
        'challenger_text': docs['late'],
    }]} for case, queries, docs in examples for side in SIDES]
    token = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    options = [token.encode(label, add_special_tokens=False) for label in 'ABC']
    if any(len(ids) != 1 for ids in options) or len({ids[0] for ids in options}) != 3:
        raise ValueError('Expected distinct one-token A, B and C decisions')
    stats = judge.score_arm(model, token, rows, 'direct', 12, 'cuda', [ids[0] for ids in options],
                            template_options=None if args.native_template else {'enable_thinking': False})
    if stats['prompts'] != 4 * len(examples):
        raise ValueError('Prompt count is inconsistent with the two-order pilot')
    values = [anchor_score(row['pairs'][0]) for row in rows]
    by_case = [[0., values[i], 0., values[i + 1]] for i in range(0, len(values), 2)]
    result = evaluate(args.label, by_case, examples)
    for i, item in enumerate(result['per_case']):
        item['two_order_predictions'] = dict(
            zip(SIDES, (r['pairs'][0]['pred'] for r in rows[2 * i:2 * i + 2])))
    report = {
        'protocol': 'Use the same four matched comparisons per case as the fixed temporal pilot. '
                    'Score the two passages with the official run\'s Direct prompt, both orders, '
                    'the selected chat template and averaged log-odds preference.',
        'model': args.model.name, 'native_template': args.native_template,
        'cases': len(examples), 'inference': stats,
        'results': {args.label: result},
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print({k: v for k, v in result.items() if k != 'per_case'}, flush=True)


if __name__ == '__main__':
    main()
