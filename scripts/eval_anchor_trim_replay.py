"""Rescore only the fixed official-dev pairs flagged by the prompt length audit."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_anchor_prompt_outliers import long_words, prompt_tokens, trim_words
from compare_anchor_reference import load
from eval_anchor_reranker import anchor_score, build, evaluate, judge
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


def make_pairs(rows, audit, tokenizer):
    expected = index(audit)
    threshold, cap = audit['words_min_characters'], audit['word_cap_counterfactual']
    affected = []
    for row in rows:
        key = row['domain'], row['qid']
        if key not in expected:
            continue
        by_id = {p['challenger']: p for p in expected[key]['pairs']}
        pairs = []
        for pair in row['pairs']:
            clipped = {side: trim_words(pair[side], threshold, cap)
                       for side in ('incumbent_text', 'challenger_text') if long_words(pair[side], threshold)}
            if not clipped:
                continue
            changed = pair | clipped
            saved = by_id.pop(pair['doc'])
            sizes = [prompt_tokens(tokenizer, row, changed, order) for order in (False, True)]
            if sizes != saved['clipped_word_tokens']:
                raise ValueError(f'Tokenization differs from the archived audit for {key}')
            pairs.append(changed)
        if by_id or not pairs:
            raise ValueError(f'Unexpected clipped document membership for {key}')
        affected.append({'domain': row['domain'], 'qid': row['qid'], 'query': row['query'], 'pairs': pairs})
    if len(affected) != len(expected):
        raise ValueError('Some archived clipped pairs are missing')
    return affected


def replay(rows, affected, raw):
    originals = index(raw)
    rescored = {(r['domain'], r['qid']): {p['doc']: anchor_score(p) for p in r['pairs']} for r in affected}
    for row in rows:
        key = row['domain'], row['qid']
        previous = originals[key]
        if previous['scores'].keys() != set(row['pool']) or previous['anchor'] != row['anchor']:
            raise ValueError(f'Candidate pool differs from the direct run for {key}')
        row['document_scores'] = previous['scores'] | rescored.get(key, {})
    _, replayed = evaluate(rows)
    paired, changes = [], []
    for current in replayed:
        key = current['domain'], current['qid']
        old = originals[key]['arms']['base_plus_anchor30']
        new = current['arms']['base_plus_anchor30']
        delta = new['ndcg'] - old['ndcg']
        paired.append((key[0], delta))
        if key in rescored:
            changes.append({'domain': key[0], 'qid': key[1], 'old': old, 'new': new,
                            'clipped_scores': rescored[key]})
        elif old != new:
            raise ValueError(f'Result outside the clipped subset changed for {key}')
    return changes, paired_delta(paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--direct-report', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output.')
    original, audit = load(args.direct_report), load(args.audit)
    rows = build(args.root, 2, set(), 100, 'dev')
    if original['model'] != args.model.name or len(rows) != original['queries'] or original['split'] != 'dev':
        raise ValueError('The model and query count must match the direct run')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = make_pairs(rows, audit, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in 'ABC']
    if any(len(t) != 1 for t in labels):
        raise ValueError('Expected one token per decision')
    stats = judge.score_arm(model, tokenizer, inputs, 'direct', 12, 'cuda', [t[0] for t in labels],
                            template_options={'enable_thinking': False})
    if (stats['prompts'], stats['prompt_tokens']) != (audit['affected_prompts'], audit['clipped_word_tokens']):
        raise ValueError('The clipped pass did not use the archived prompt budget')
    changes, delta = replay(rows, inputs, original)
    base = original['summary']['base_plus_anchor30']['ndcg']
    summary = {
        'direct_ndcg': base, 'clipped_replay_ndcg': base + delta['point'], 'paired_delta': delta,
        'changed_top_ten': sum(r['old']['ranking'] != r['new']['ranking'] for r in changes),
    }
    report = {
        'protocol': 'Score the 84 clipped prompts with the same model and options. Combine their updated '
                    'scores with the cached direct scores and recompute the fixed fusion. The clipped prompts '
                    'are batched separately. Candidates and all other document scores come from the direct run.',
        'inference': stats, 'summary': summary, 'per_query': changes,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
