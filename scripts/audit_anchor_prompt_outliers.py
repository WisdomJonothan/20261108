"""Check sparse token-length outliers under the frozen anchor prompt protocol."""
import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata

from transformers import AutoTokenizer

from eval_anchor_reranker import build, judge


def long_words(text, minimum):
    return [word for word in text.split() if len(word) >= minimum]


def describe(words):
    longest = max(words, key=len)
    counts = Counter(longest)
    char, frequency = counts.most_common(1)[0]
    return {
        'words_over_threshold': len(words), 'longest_word_characters': len(longest),
        'distinct_codepoints_in_longest_word': len(counts),
        'most_common_codepoint_category': unicodedata.category(char),
        'most_common_codepoint_share': frequency / len(longest),
    }


def trim_words(text, threshold, cap):
    return ' '.join(word[:cap] if len(word) >= threshold else word for word in text.split())


def prompt_tokens(tokenizer, row, pair, reverse):
    content = judge.prompt(row, pair, False, reverse)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def audit(rows, tokenizer, threshold, cap):
    outliers = []
    for row in rows:
        details, affected = {}, []
        for pair in row['pairs']:
            words = {}
            for side in ('incumbent', 'challenger'):
                text = pair[side + '_text']
                doc = row['anchor'] if side == 'incumbent' else pair['doc']
                if doc not in details:
                    found = long_words(text, threshold)
                    details[doc] = describe(found) if found else None
                if details[doc]:
                    words[side] = trim_words(text, threshold, cap)
            if not words:
                continue
            alternative = pair | {side + '_text': value for side, value in words.items()}
            tokens = [prompt_tokens(tokenizer, row, pair, order) for order in (False, True)]
            compact = [prompt_tokens(tokenizer, row, alternative, order) for order in (False, True)]
            affected.append({'challenger': pair['doc'], 'original_tokens': tokens,
                             'clipped_word_tokens': compact})
        if affected:
            outliers.append({
                'domain': row['domain'], 'qid': row['qid'],
                'oversized_windows': {doc: item for doc, item in details.items() if item},
                'pairs': affected,
            })
    return outliers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=int, default=2048)
    parser.add_argument('--cap', type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.cap < args.threshold or args.output.exists():
        parser.error('Supply a positive cap below the threshold and a new output.')
    rows = build(args.root, 2, set(), 100, 'dev')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    outliers = audit(rows, tokenizer, args.threshold, args.cap)
    pairs = [p for row in outliers for p in row['pairs']]
    report = {
        'protocol': 'Scan every selected passage in the fixed official dev depth-100 run. '
                    'Measure both original and clipped prompt token lengths when a selected passage '
                    'contains an oversized whitespace-delimited word. This report uses tokenization.',
        'queries': len(rows), 'words_min_characters': args.threshold, 'word_cap_counterfactual': args.cap,
        'outlier_queries': len(outliers), 'outlier_windows': sum(len(r['oversized_windows']) for r in outliers),
        'affected_prompts': 2 * len(pairs),
        'original_tokens': sum(sum(pair['original_tokens']) for pair in pairs),
        'clipped_word_tokens': sum(sum(pair['clipped_word_tokens']) for pair in pairs),
        'maximum_original': max((max(p['original_tokens']) for p in pairs), default=0),
        'maximum_clipped': max((max(p['clipped_word_tokens']) for p in pairs), default=0),
        'per_query': outliers,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print({name: value for name, value in report.items() if name != 'per_query'}, flush=True)


if __name__ == '__main__':
    main()
