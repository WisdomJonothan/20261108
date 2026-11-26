"""Measure whether a fixed evidence pair flips when the temporal condition changes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import time

from compare_anchor_reference import load
from coverage_cf_bound import read_qrels


SIDES = ('early', 'late')


def prepare(manifest, data_root):
    needed = defaultdict(set)
    for case in manifest:
        needed[case['domain']].update(e['doc'] for e in case['evidence'].values())
    corpus, labels = {}, {}
    for domain, ids in needed.items():
        labels[domain] = read_qrels(data_root / domain / 'qrels_train.txt')
        with (data_root / domain / 'documents.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if record['id'] in ids:
                    corpus[record['id']] = ' '.join(record['content'].split())
                    if ids <= corpus.keys():
                        break
    examples = []
    for case in manifest:
        queries = {side: case['query_template'].format(time=case['time_values'][side]) for side in SIDES}
        gold = labels[case['domain']][case['source_train_qid']]
        if queries['early'] == queries['late'] or case['query_template'].count('{time}') != 1:
            raise ValueError(f'Expected exactly one varying temporal condition for {case["id"]}')
        passages = {}
        for side in SIDES:
            item = case['evidence'][side]
            if item['doc'] not in gold:
                raise ValueError(f'Source document is outside the labeled training topic for {case["id"]}')
            passage = corpus[item['doc']][item['start']:item['end']]
            if hashlib.sha256(passage.encode('utf-8')).hexdigest() != item['sha256']:
                raise ValueError(f'Source text changed for {case["id"]}/{side}')
            passages[side] = passage
        examples.append((case, queries, passages))
    return examples


def overlap(query, passage):
    q = set(re.findall(r'[a-z0-9]+', query.lower()))
    d = set(re.findall(r'[a-z0-9]+', passage.lower()))
    return len(q & d) / len(q)


def evaluate(name, model_values, examples):
    rows = []
    for (case, _, _), values in zip(examples, model_values):
        by_query = {q: dict(zip(SIDES, values[i * 2:i * 2 + 2])) for i, q in enumerate(SIDES)}
        margins = {q: by_query[q][q] - by_query[q][other]
                   for q, other in (('early', 'late'), ('late', 'early'))}
        outcomes = {q: 'correct' if margin > 0 else 'incorrect' if margin < 0 else 'tie'
                    for q, margin in margins.items()}
        rows.append({'case': case['id'], 'domain': case['domain'], 'scores': by_query,
                     'correct_document_margins': margins, 'outcomes': outcomes})
    if len(rows) != len(examples):
        raise ValueError(f'Scoring results are incomplete for {name}')
    return {
        'query_variants': 2 * len(rows),
        'correct_variants': sum(o == 'correct' for r in rows for o in r['outcomes'].values()),
        'tied_variants': sum(o == 'tie' for r in rows for o in r['outcomes'].values()),
        'correct_flips': sum(set(r['outcomes'].values()) == {'correct'} for r in rows),
        'reversed_flips': sum(set(r['outcomes'].values()) == {'incorrect'} for r in rows),
        'mean_temporal_margin_shift': mean(sum(r['correct_document_margins'].values()) for r in rows),
        'per_case': rows,
    }


def cross_encoder(path, pairs, threads, *, device='cpu', precision='float32'):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    token = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, dtype=getattr(torch, precision)).to(device).eval()
    values = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(pairs), 8):
            chunk = pairs[start:start + 8]
            batch = token([q for q, _ in chunk], [d for _, d in chunk], padding=True, return_tensors='pt')
            if batch['input_ids'].shape[1] > 512:
                raise ValueError('A diagnostic input exceeds the cross-encoder token budget')
            logits = model(**batch.to(device)).logits.float()
            if logits.shape != (len(chunk), 1):
                raise ValueError('Expected one raw score per example')
            values.extend(logits[:, 0].tolist())
    return values, time.perf_counter() - started


def qwen_reranker(path, pairs, threads, dated_pairs=None):
    from run_qwen3_rerank import INSTRUCT

    instruction = ('Identify whether the passage directly supports the event, record, or state '
                   'under the query time condition. Apply the condition to the date of the event, '
                   'record, or state being asked about. Distinguish it from unrelated dates and '
                   'the date of a later discussion or translation.')
    instructions = {'qwen_default': INSTRUCT, 'qwen_event_time': instruction}
    variants = {name: pairs for name in instructions}
    if dated_pairs is not None:
        instructions = dict.fromkeys(dated_pairs, instruction)
        variants = dated_pairs
    return score_qwen_variants(path, variants, instructions, threads), instructions


def score_qwen_variants(path, variants, instructions, threads, *, device='cpu', precision='float32'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from run_qwen3_rerank import PREFIX, SUFFIX

    torch.set_num_threads(threads)
    token = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=getattr(torch, precision)).to(device).eval()
    prefix = token.encode(PREFIX, add_special_tokens=False)
    suffix = token.encode(SUFFIX, add_special_tokens=False)
    answers = [token.encode(answer, add_special_tokens=False) for answer in ('no', 'yes')]
    if not all(len(ids) == 1 for ids in answers) or answers[0] == answers[1]:
        raise ValueError('Expected distinct single-token Qwen relevance answers')
    no, yes = (ids[0] for ids in answers)
    results = {}
    with torch.inference_mode():
        for name, task in instructions.items():
            pairs = variants[name]
            started = time.perf_counter()
            values = []
            for start in range(0, len(pairs), 8):
                chunk = pairs[start:start + 8]
                bodies = [f'<Instruct>: {task}\n<Query>: {q}\n<Document>: {d}' for q, d in chunk]
                ids = token(bodies, add_special_tokens=False)['input_ids']
                batch = token.pad({'input_ids': [prefix + x + suffix for x in ids]},
                                  padding=True, return_tensors='pt')
                logits = model(**batch.to(device), logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
                values.extend((logits[:, yes] - logits[:, no]).tolist())
            results[name] = (values, time.perf_counter() - started)
    return results



def add_event_dates(examples, annotations):
    if annotations.keys() != {case['id'] for case, _, _ in examples}:
        raise ValueError('Event date annotations must cover every diagnostic case')
    variants = {'qwen_matched_dates': [], 'qwen_exchanged_dates': []}
    for case, queries, docs in examples:
        dates = annotations[case['id']]
        if dates.keys() != set(SIDES) or not all(dates.values()):
            raise ValueError(f'Missing event date for {case["id"]}')
        for query in (queries[q] for q in SIDES):
            for side, other in (('early', 'late'), ('late', 'early')):
                variants['qwen_matched_dates'].append((query, docs[side] + '\nEvent time: ' + dates[side]))
                variants['qwen_exchanged_dates'].append((query, docs[side] + '\nEvent time: ' + dates[other]))
    return variants



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--zero-shot', type=Path)
    parser.add_argument('--frozen-ce', type=Path)
    parser.add_argument('--qwen-reranker', type=Path)
    parser.add_argument('--annotated-event-dates', type=Path,
                        help='Compare matched and exchanged event dates with the explicit Qwen instruction.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error('Use a positive thread count and a new output.')
    if bool(args.zero_shot) != bool(args.frozen_ce) or not (args.zero_shot or args.qwen_reranker):
        parser.error('Supply the two BGE checkpoints, a Qwen reranker, or both.')
    if args.annotated_event_dates and not args.qwen_reranker:
        parser.error('The supplied-date probe needs a Qwen reranker.')
    examples = prepare(load(args.manifest)['cases'], args.data_root)
    pairs = [[(queries[q], docs[d]) for q in SIDES for d in SIDES]
             for _, queries, docs in examples]
    flat = [item for group in pairs for item in group]
    results = {'overlap': evaluate('overlap', [[overlap(q, d) for q, d in group] for group in pairs], examples)}
    raw, model_paths, instructions = {}, {}, {}
    if args.zero_shot:
        for name, path in (('bge_zero_shot', args.zero_shot), ('bge_frozen_ce', args.frozen_ce)):
            raw[name] = cross_encoder(path, flat, args.threads)
        model_paths.update(bge_zero_shot=args.zero_shot.name, bge_frozen_ce=args.frozen_ce.parent.name)
    if args.qwen_reranker:
        variants = (add_event_dates(examples, load(args.annotated_event_dates)['times'])
                    if args.annotated_event_dates else None)
        extra, instructions = qwen_reranker(args.qwen_reranker, flat, args.threads, variants)
        raw.update(extra)
        model_paths.update({name: args.qwen_reranker.name for name in extra})
    for name, (values, seconds) in raw.items():
        if len(values) != len(flat):
            raise ValueError(f'Expected every query-document score for {name}')
        results[name] = evaluate(name, [values[i:i + 4] for i in range(0, len(values), 4)], examples)
        results[name]['scoring_seconds'] = seconds
        print(name, {k: v for k, v in results[name].items() if k != 'per_case'}, flush=True)
    report = {
        'protocol': 'Compare the same two exact source passages under both query conditions. '
                    'Both source documents are official positives for the originating train topic. '
                    'A correct flip requires the expected passage to rank strictly first for both variants.',
        'cases': len(examples), 'domains': len({case['domain'] for case, _, _ in examples}),
        'model_paths': model_paths, 'qwen_instructions': instructions,
        'results': results,
    }
    if args.annotated_event_dates:
        report['event_date_annotations'] = args.annotated_event_dates.name
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print('overlap', {k: v for k, v in results['overlap'].items() if k != 'per_case'})


if __name__ == '__main__':
    main()
