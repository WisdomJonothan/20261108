"""Select evidence with a frozen CE and replay the same six-pair judge."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

import benchmark_pairwise_replacement_v2 as judge
from replay_position_discount import bootstrap, eligible


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build(root, fold, sample):
    split = 'dev' if fold == 'dev' else 'train'
    scope = 'dev' if fold == 'dev' else f'train_{fold}'
    directory = root / 'results/replacement_v2'
    manifest = load(directory / f'candidates_{scope}.json')['queries']
    cache = load(directory / f'expanded3_fixed_{scope}.json')['budgets']['18']['per_query']
    history = {(r['domain'], r['qid']): r for r in cache}
    assert len(manifest) == len(history)
    rows = []
    for domain in judge.DOMAINS:
        subset = sorted((r for r in manifest if r['domain'] == domain), key=lambda r: r['qid'])
        if sample and len(subset) > sample:
            # Deterministic across domains. No qrels are consulted for the sample.
            subset = [subset[int(i * len(subset) / sample)] for i in range(sample)]
        domain_rows = []
        for r in subset:
            old = history[domain, r['qid']]
            actions = sorted(old['actions'], key=lambda a: (a['slot'], -a['prior']))[:6]
            assert len(actions) == 6
            domain_rows.append({'domain': domain, 'qid': r['qid'],
                                'query': judge.strip_html(r['query']),
                                'baseline': old['baseline'], 'actions': actions})
        need = {a[k] for r in domain_rows for a in r['actions'] for k in ('incumbent', 'challenger')}
        corpus = {}
        if need:
            path = root / 'data/track1_tempo' / domain / 'documents.jsonl'
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    doc = json.loads(line)
                    if doc['id'] in need:
                        corpus[doc['id']] = doc['content']
                        if len(corpus) == len(need):
                            break
        assert set(corpus) == need
        for r in domain_rows:
            terms = judge.terms(r['query'])
            docs = {a[k] for a in r['actions'] for k in ('incumbent', 'challenger')}
            r['windows'] = {doc: judge.passages(corpus[doc], terms)[:8] for doc in sorted(docs)}
        rows.extend(domain_rows)
    return rows


@torch.inference_mode()
def choose_windows(rows, model, tokenizer):
    count = 0
    for r in rows:
        query_ids = tokenizer(r['query'], add_special_tokens=False, truncation=True,
                              max_length=192)['input_ids']
        query = tokenizer.decode(query_ids, skip_special_tokens=True)
        items = [(doc, window) for doc, windows in r['windows'].items() for window in windows]
        selected = {}
        for head in range(0, len(items), 32):
            batch = items[head:head + 32]
            inp = tokenizer([query] * len(batch), [p for _, p in batch], padding=True,
                            truncation=True, max_length=384, return_tensors='pt').to('cuda')
            scores = model(**inp).logits.view(-1).float().cpu().tolist()
            for (doc, passage), score in zip(batch, scores):
                if not math.isfinite(score):
                    raise ValueError(f'Nonfinite CE score for {doc}')
                if doc not in selected or selected[doc][0] < score:
                    selected[doc] = (score, passage)
        r['pairs'] = []
        for i, action in enumerate(r['actions']):
            inc, ch = action['incumbent'], action['challenger']
            left, right = selected[inc][1], selected[ch][1]
            if (left, right) != (r['windows'][inc][0], r['windows'][ch][0]):
                r['pairs'].append({'index': i, 'incumbent_text': left, 'challenger_text': right})
        count += len(items)
    return count


def decision(actions):
    return max(eligible(actions), key=lambda a: a['mp'], default=None)


def summarize(rows):
    changes, details, counts = [], [], {'lexical': Counter(), 'semantic': Counter()}
    for row in rows:
        original = row['actions']
        changed = [a.copy() for a in original]
        for item in row['pairs']:
            votes = [item['pred'][order] for order in ('fwd', 'rev')]
            change = changed[item['index']]
            change['p_fwd'], change['p_rev'] = [v['challenger'] for v in votes]
            change['mp'] = (change['p_fwd'] + change['p_rev']) / 2
            change['agree'] = all(max(v, key=v.get) == 'challenger' for v in votes)
        entry = {k: row[k] for k in ('qid', 'domain', 'baseline')}
        entry['changed_pairs'] = len(row['pairs'])
        for key, ac in [('lexical', original), ('semantic', changed)]:
            selected = decision(ac)
            delta = selected['delta'] if selected else 0.
            beneficial = any(a['delta'] > 1e-12 for a in eligible(ac))
            counts[key]['has_accepted_benefit'] += beneficial
            counts[key]['kept' if selected is None else 'helpful' if delta > 1e-12
                        else 'harmful' if delta < -1e-12 else 'neutral'] += 1
            entry[key] = {'score': row['baseline'] + delta, 'delta': delta, 'chosen': selected}
        changes.append((row['domain'], entry['semantic']['score'] - entry['lexical']['score']))
        entry['rescored_actions'] = [changed[p['index']] for p in row['pairs']]
        details.append(entry)
    scores = {key: bootstrap([(r['domain'], r[key]['score']) for r in details])['point']
              for key in counts}
    return {'queries': len(rows), 'domains': len({r['domain'] for r in rows}),
            'scores': scores, 'semantic_vs_lexical': bootstrap(changes), 'counts': counts,
            'changed_pairs': sum(r['changed_pairs'] for r in details)}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--selector', type=Path, required=True)
    parser.add_argument('--judge', type=Path, required=True)
    parser.add_argument('--fold', choices=('f0', 'f1', 'f2', 'dev'), required=True)
    parser.add_argument('--sample-per-domain', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output exists: {args.output}')
    rows = build(args.root, args.fold, args.sample_per_domain)
    selector_tokenizer = AutoTokenizer.from_pretrained(args.selector, local_files_only=True)
    selector = AutoModelForSequenceClassification.from_pretrained(
        args.selector, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    print('Selection queries', len(rows), flush=True)
    ce_pairs = choose_windows(rows, selector, selector_tokenizer)
    del selector
    torch.cuda.empty_cache()
    rescored = [r for r in rows if r['pairs']]
    stats = {}
    if rescored:
        tokenizer = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.judge, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
        option_ids = [tokenizer.encode(s, add_special_tokens=False) for s in 'ABC']
        assert all(len(ids) == 1 for ids in option_ids)
        print('Rescoring changed pairs', sum(len(r['pairs']) for r in rescored), flush=True)
        stats = judge.score_arm(model, tokenizer, rescored, 'direct', 12, 'cuda',
                                [ids[0] for ids in option_ids])
    summary, details = summarize(rows)
    report = {'protocol': 'Frozen six head-first pairs and acceptance. Select one up-to-112-word '
              'window per document from the lexical top eight via frozen BGE CE. '
              'The same Qwen Direct prompt judges both orders. Pairs whose passages are identical '
              'reuse their cached original scores. Labels are used only through cached deltas.',
              'fold': args.fold, 'sample_per_domain': args.sample_per_domain,
              'selection_pairs': ce_pairs, 'judge_inference': stats,
              'summary': summary, 'per_query': details}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
