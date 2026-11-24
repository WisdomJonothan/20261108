"""Change only the judge checkpoint on the originally routed Direct comparisons."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from eval_abstention_shortlist import outcome
from eval_joint_swap_support import prepare, summarize
from eval_swap_context_extent import build_cases
from replay_position_discount import eligible


POLICIES = ('external6', 'direct', 'teacher')


def key(case):
    return case['domain'], case['qid'], case['incumbent'], case['challenger']


def judge(cases, checkpoint, batch_size, disable_thinking):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark_pairwise_replacement_v2 import score_arm

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in 'ABC']
    if not all(len(token) == 1 for token in labels) or len(set(t[0] for t in labels)) != 3:
        raise ValueError('Expected distinct single-token A, B and C answers')
    groups = defaultdict(list)
    for case in cases:
        groups[case['domain'], case['qid']].append((case, case['variants']['short'].copy()))
    rows = [{'query': group[0][0]['query'], 'pairs': [pair for _, pair in group]}
            for group in groups.values()]
    options = {'enable_thinking': False} if disable_thinking else {}
    stats = score_arm(model, tokenizer, rows, 'direct', batch_size, 'cuda', [t[0] for t in labels],
                      template_options=options)
    if stats['prompts'] != 2 * len(cases):
        raise ValueError('Prompt count disagrees with the fixed two-order comparison')
    for group in groups.values():
        for case, pair in group:
            case['predictions']['teacher'] = pair['pred']
    return stats


def rerank(cases, checkpoint, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_qwen3_rerank import INSTRUCT, PREFIX, SUFFIX

    documents = {}
    for case in cases:
        for side in ('incumbent', 'challenger'):
            key = case['domain'], case['qid'], case[side]
            documents[key] = case['query'], case['variants']['short'][side + '_text']
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in ('no', 'yes')]
    if not all(len(token) == 1 for token in labels) or labels[0] == labels[1]:
        raise ValueError('Expected distinct single-token no and yes answers')
    scores = {}
    tokens = 0
    records = list(documents.items())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            bodies = [f'<Instruct>: {INSTRUCT}\n<Query>: {q}\n<Document>: {d}' for _, (q, d) in chunk]
            encoded = tokenizer(bodies, add_special_tokens=False)['input_ids']
            if any(len(prefix) + len(body) + len(suffix) > 4096 for body in encoded):
                raise ValueError('An original excerpt exceeds the reranker input budget')
            batch = tokenizer.pad({'input_ids': [prefix + body + suffix for body in encoded]},
                                  padding=True, return_tensors='pt').to(model.device)
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            p_yes = logits[:, [t[0] for t in labels]].float().softmax(-1)[:, 1].cpu().tolist()
            scores.update((key, p) for (key, _), p in zip(chunk, p_yes))
            tokens += int(batch['attention_mask'].sum().item())
    for case in cases:
        case['predictions']['document_support'] = {
            side: scores[case['domain'], case['qid'], case[side]] for side in ('incumbent', 'challenger')
        }
    return {'prompts': len(documents), 'prompt_tokens': tokens, 'seconds_total': time.perf_counter() - started,
            'gpu_peak_mb': torch.cuda.max_memory_allocated() / 2**20}


def decision(case):
    if 'document_support' in case['predictions']:
        support = case['predictions']['document_support']
        gap = support['challenger'] - support['incumbent']
        return {'mp': .5 + gap / 2, 'agree': gap > 0, 'support_gap': gap}
    votes = case['predictions']['teacher']
    fwd, rev = votes['fwd']['challenger'], votes['rev']['challenger']
    agree = all(max(vote, key=vote.get) == 'challenger' for vote in votes.values())
    return {'p_fwd': fwd, 'p_rev': rev, 'mp': mean((fwd, rev)), 'agree': agree}


def diagnostic(cases, protocol):
    from scipy.stats import rankdata

    results = {}
    for label in ('challenger', 'incumbent', 'both', 'neither'):
        rows = [case for case in cases if case['gold_outcome'] == label]
        if not rows:
            continue
        decisions = [decision(case) for case in rows]
        passed = [bool(eligible([d])) for d in decisions]
        results[label] = {
            'pairs': len(rows),
            'direct_accepted': sum(case['historical_accepted'] for case in rows),
            'teacher_accepted': sum(passed),
            'new_accepted': sum(not case['historical_accepted'] and accept for case, accept in zip(rows, passed)),
            'lost_accepted': sum(case['historical_accepted'] and not accept for case, accept in zip(rows, passed)),
        }
        if protocol == 'direct':
            results[label]['teacher_one_order_above_half'] = sum(
                (d['p_fwd'] > .5) != (d['p_rev'] > .5) for d in decisions)
    exclusive = [case for case in cases if case['gold_outcome'] in ('challenger', 'incumbent')]
    positives = [i for i, case in enumerate(exclusive) if case['gold_outcome'] == 'challenger']
    p, n = len(positives), len(exclusive) - len(positives)
    if p and n:
        def auc(scores):
            ranks = rankdata(scores)
            return float((sum(ranks[i] for i in positives) - p * (p + 1) / 2) / (p * n))

        variation = ('teacher_worst_order' if protocol == 'direct' else 'teacher_document_score_gap')
        teacher_scores = (
            [min(decision(case)['p_fwd'], decision(case)['p_rev']) for case in exclusive]
            if protocol == 'direct' else [decision(case)['support_gap'] for case in exclusive]
        )
        results['exclusive_pair_auc'] = {
            'direct_worst_order': auc([min(case['historical_probabilities']) for case in exclusive]),
            variation: auc(teacher_scores),
        }
    return results


def full_results(rows, cases):
    matched = {key(case): case for case in cases}
    results = []
    for row in rows:
        direct = row['arms']['supervised']
        arms = {'external6': outcome(row['arms']['external6'], True), 'direct': outcome(direct, True)}
        suffix = []
        judgments = []
        for action, labels in zip(direct[3:], row['treatment']):
            case = matched.pop((row['domain'], row['qid'], action['incumbent'], action['challenger']))
            teacher = {name: action[name] for name in ('slot', 'incumbent', 'challenger', 'delta')} | decision(case)
            suffix.append(teacher)
            judgments.append(labels | {'accepted': {'direct': bool(eligible([action])),
                                                    'teacher': bool(eligible([teacher]))}})
        arms['teacher'] = outcome(direct[:3] + suffix if suffix else direct, True)
        results.append({'domain': row['domain'], 'qid': row['qid'], 'fold': row['fold'], 'baseline': row['baseline'],
                        'routed': row['routed'], 'outcomes': arms, 'support_judgments': judgments})
    if matched:
        raise ValueError('Stored decisions contain unmatched pairs')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--direct-report', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--protocol', choices=('direct', 'reranker'), default='direct')
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--full-fold', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    data = args.data_root or args.root / 'data/track1_tempo'
    reference = load(args.direct_report)
    cases = build_cases(args.root, data, reference, exclusive_only=not args.full_fold)
    if args.protocol == 'direct':
        stats = judge(cases, args.model, args.batch_size, args.disable_thinking)
        scoring = 'Use the same Direct prompt and two-order acceptance gate with a different checkpoint.'
    else:
        stats = rerank(cases, args.model, args.batch_size)
        scoring = 'Score each distinct document using the Qwen3-Reranker yes/no logit protocol. ' \
                  'Accept a positive challenger-minus-incumbent probability difference.'
    result = {
        'protocol': 'Keep the Direct question, original 112-word snippets and exact pairs. '
                    + scoring + ' The judge changes only in the routed supervised suffix. The '
                    'pilot includes every suffix pair with exactly one positive qrel. The full-fold '
                    'protocol scores every suffix pair and preserves the prefix and routing.',
        'model': args.model_label, 'dtype': 'bfloat16', 'disable_thinking': args.disable_thinking,
        'scoring_protocol': args.protocol,
        'folds': reference['folds'], 'inference': stats, 'pair_diagnostics': diagnostic(cases, args.protocol),
        'pairs': [{k: v for k, v in case.items() if k not in ('query', 'variants')} for case in cases],
    }
    if args.full_fold:
        rows = full_results(prepare(args.root, data, reference, reference['folds']), cases)
        result['summary'] = summarize(rows, POLICIES)
        result['per_query'] = rows
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ('pairs', 'per_query')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
