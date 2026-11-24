"""Counterfactual context expansion on fixed beneficial and harmful swap pairs."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from audit_rejected_swap_evidence import read_documents
from compare_anchor_reference import load
from eval_joint_swap_support import prepare
from replay_position_discount import eligible


WINDOWS = ('short', 'challenger_long', 'both_long')
CAP = 384


def expand(clean, excerpt):
    words = clean.split()
    if len(words) <= CAP:
        return ' '.join(words)
    width = len(excerpt.split())
    center = next(index for index in range(0, len(words), 80)
                  if ' '.join(words[index:index + width]) == excerpt)
    start = min(max(center - (CAP - width) // 2, 0), len(words) - CAP)
    return ' '.join(words[start:start + CAP])


def build_cases(root, data_root, direct, *, exclusive_only=True):
    from benchmark_pairwise_replacement_v2 import passages, strip_html, terms

    by_domain = defaultdict(list)
    for row in prepare(root, data_root, direct, direct['folds']):
        for action, labels in zip(row['arms']['supervised'][3:], row['treatment']):
            if not exclusive_only or labels['gold_outcome'] in ('incumbent', 'challenger'):
                by_domain[row['domain']].append((row, action, labels['gold_outcome']))
    cases = []
    for domain, entries in sorted(by_domain.items()):
        wanted = {action[side] for _, action, _ in entries for side in ('incumbent', 'challenger')}
        docs = read_documents(data_root / domain / 'documents.jsonl', wanted)
        for row, action, outcome in entries:
            query = strip_html(row['query'])
            query_terms = terms(query)
            short = {side + '_text': passages(docs[action[side]], query_terms)[0]
                     for side in ('incumbent', 'challenger')}
            long = {side + '_text': expand(strip_html(docs[action[side]]), short[side + '_text'])
                    for side in ('incumbent', 'challenger')}
            variants = {'short': short, 'challenger_long': short | {'challenger_text': long['challenger_text']},
                        'both_long': long}
            cases.append({
                'domain': domain, 'qid': row['qid'], 'query': query,
                'incumbent': action['incumbent'], 'challenger': action['challenger'],
                'gold_outcome': outcome, 'delta': action['delta'],
                'historical_accepted': bool(eligible([action])),
                'historical_probabilities': [action['p_fwd'], action['p_rev']],
                'variants': variants, 'predictions': {},
            })
    return cases


def score(cases, checkpoint, variants=WINDOWS):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark_pairwise_replacement_v2 import score_arm

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    token = [tokenizer.encode(label, add_special_tokens=False) for label in 'ABC']
    if not all(len(t) == 1 for t in token):
        raise ValueError('Expected one token per output label')
    stats = {}
    for variant in variants:
        grouped = defaultdict(list)
        for case in cases:
            pair = case['variants'][variant]
            identical = next((previous for previous in case['predictions']
                              if case['variants'][previous] == pair), None)
            if identical is not None:
                case['predictions'][variant] = case['predictions'][identical]
                continue
            grouped[case['domain'], case['qid']].append((case, pair.copy()))
        rows = [{'query': entries[0][0]['query'], 'pairs': [pair for _, pair in entries]}
                for entries in grouped.values()]
        stats[variant] = score_arm(model, tokenizer, rows, 'direct', 12, 'cuda', [t[0] for t in token])
        for entries in grouped.values():
            for case, pair in entries:
                vote = pair['pred']
                p = [vote[order]['challenger'] for order in ('fwd', 'rev')]
                agree = all(max(v, key=v.get) == 'challenger' for v in vote.values())
                case['predictions'][variant] = {'accepted': bool(agree and mean(p) > .5), 'vote': vote}
        print(variant, 'fresh prompts:', stats[variant]['prompts'], flush=True)
    return stats


def summarize(records):
    output = {}
    for label in ('challenger', 'incumbent'):
        rows = [r for r in records if r['gold_outcome'] == label]
        counts = Counter()
        for row in rows:
            original = row['predictions']['short']['accepted']
            for name, pred in row['predictions'].items():
                shifted = pred['accepted']
                counts[name, 'accepted'] += shifted
                counts[name, 'newly_accepted'] += shifted and not original
                counts[name, 'no_longer_accepted'] += original and not shifted
        output[label] = {
            'pairs': len(rows), 'historical_accepted': sum(row['historical_accepted'] for row in rows),
            'historical_gate_agreement': sum(row['historical_accepted'] == row['predictions']['short']['accepted']
                                             for row in rows),
            'variants': {name: {key: counts[name, key] for key in
                               ('accepted', 'newly_accepted', 'no_longer_accepted')}
                         for name in WINDOWS},
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--direct-report', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    data_root = args.data_root or args.root / 'data/track1_tempo'
    control = load(args.direct_report)
    cases = build_cases(args.root, data_root, control)
    stats = score(cases, args.model)
    records = []
    for case in cases:
        lengths = {name: {side: len(text.split()) for side, text in v.items()}
                   for name, v in case['variants'].items()}
        records.append({k: v for k, v in case.items() if k not in ('query', 'variants')} | {'words': lengths})
    report = {
        'protocol': 'Qrels select all suffix pairs with exactly one positive document for a diagnostic. '
                    'Replay original Direct with both presentation orders. Expand the challenger alone '
                    'or both documents to 384 words around the original excerpt. Preserve the model, '
                    'query, documents and acceptance rule. Exact duplicate prompts reuse scores.',
        'folds': control['folds'], 'inference': stats, 'summary': summarize(records), 'pairs': records,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(report['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
