"""Compare the native Direct judge with the same checkpoint scoring passages alone."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmark_pairwise_replacement_v2 as judge
from compare_anchor_reference import load
from eval_temporal_query_contrast import SIDES, evaluate, prepare


def score(model, tokenizer, rows, yes, no):
    tokenizer.padding_side = 'left'
    log_odds, probabilities = [], []
    prompt_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            chats = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': judge.perdoc_prompt(row['query'], row['doc_text'][side])}],
                tokenize=False, add_generation_prompt=True) for side in SIDES]
            batch = tokenizer(chats, add_special_tokens=False, padding=True, return_tensors='pt').to(model.device)
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
            odds = lp[:, yes] - lp[:, no]
            log_odds.extend(odds.cpu().tolist())
            probabilities.extend(odds.sigmoid().cpu().tolist())
            prompt_tokens += int(batch['attention_mask'].sum().item())
    stats = {'prompts': len(log_odds), 'prompt_tokens': prompt_tokens,
             'seconds_total': time.perf_counter() - started,
             'gpu_peak_mb': torch.cuda.max_memory_allocated() / 2**20}
    return stats, log_odds, probabilities



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output.')
    examples = prepare(load(args.manifest)['cases'], args.data_root)
    rows = [{'query': queries[side], 'top': list(SIDES), 'pairs': [], 'doc_text': docs}
            for _, queries, docs in examples for side in SIDES]
    token = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    yes, no = [token.encode(answer, add_special_tokens=False) for answer in ('yes', 'no')]
    if len(yes) != 1 or len(no) != 1 or yes == no:
        raise ValueError('Expected distinct one-token yes and no decisions')
    stats, log_odds, probabilities = score(model, token, rows, yes[0], no[0])
    if stats['prompts'] != 4 * len(examples):
        raise ValueError('Pointwise run must score each passage for both query variants')
    scored = {'pointwise_qwen4b': probabilities, 'pointwise_qwen4b_logodds': log_odds}
    results = {name: evaluate(name, [values[i:i + 4] for i in range(0, len(values), 4)], examples)
               for name, values in scored.items()}
    report = {
        'protocol': 'Use the same Qwen checkpoint and native template to score each passage alone. '
                    'Reuse the temporal queries, exact passage spans and 44-prompt budget.',
        'model': args.model.name, 'cases': len(examples), 'inference': stats,
        'results': results,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for name, result in results.items():
        print(name, {k: v for k, v in result.items() if k != 'per_case'}, flush=True)


if __name__ == '__main__':
    main()
