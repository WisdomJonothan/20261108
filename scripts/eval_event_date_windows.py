"""Evaluate fixed-budget date-centered excerpts on disjoint RETECO training topics."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean
import tarfile

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import strip_html, terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_temporal_query_contrast import score_qwen_variants
from replay_anchor_cap import z
from replay_anchor_edit_scope import paired_delta
from run_qwen3_rerank import INSTRUCT


YEAR = re.compile(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b')
TIME = re.compile(r'\b(?:before|after|since|until|prior to|between|from|by|as of|during|in|around|through)'
                  r'\s+(?:(?:early|mid|late)[- ]+)?(?:1[0-9]{3}|20[0-2][0-9])\b', re.I)
SENTENCE = re.compile(r'(?<=[.!?])\s+|[\r\n]+')
ARMS = ('lexical', 'date_only', 'linked_date')
WINDOW = 96
POOL = 14


def key_terms(text):
    return terms(YEAR.sub(' ', text))


def comparison_queries(archive, manifest):
    source_cases = {(r['domain'], r['source_train_qid']) for r in load(manifest)['cases']}
    panels = set()
    with tarfile.open(archive) as stream:
        for member in stream:
            payload = json.load(stream.extractfile(member))
            panels.update((r['domain'], r['qid']) for r in payload['per_query'])
    return panels, source_cases


def sample(data_root, runs, panels, source_cases, limit, transfer):
    groups = {}
    for path in sorted(data_root.glob('*/examples_train.jsonl')):
        domain = path.parent.name
        if transfer and domain == 'history':
            continue
        group = []
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                qid = row['id']
                key = (domain, qid)
                if key in source_cases or (key in panels) != transfer:
                    continue
                text = strip_html(row['query'])
                query = ' '.join(text.split())
                focus = [key_terms(s) for s in SENTENCE.split(text) if TIME.search(s)]
                focus = [clause for clause in focus if len(clause) >= 2]
                if focus:
                    group.append({'qid': qid, 'query': query, 'focus': focus})
        if not group:
            continue
        group.sort(key=lambda r: hashlib.sha256(f'{domain}/{r["qid"]}'.encode()).digest())
        chosen = {r['qid']: r | {'pool': []} for r in (group[:limit] if limit else group)}
        with (runs / f'dense_mxbai_{domain}_train.trec').open(encoding='utf-8') as stream:
            for line in stream:
                qid, _, doc, rank, score, _ = line.split()
                if qid in chosen and int(rank) <= POOL:
                    chosen[qid]['pool'].append((int(rank), doc, float(score)))
        for row in chosen.values():
            row['pool'].sort()
            if [rank for rank, _, _ in row['pool']] != list(range(1, POOL + 1)):
                raise ValueError(f'Incomplete dense pool for {domain} {row["qid"]}')
            if len({doc for _, doc, _ in row['pool']}) != POOL:
                raise ValueError(f'Duplicate dense documents for {domain} {row["qid"]}')
        groups[domain] = list(chosen.values())
    return groups


def excerpts(body, vocabulary, focus):
    sentences = SENTENCE.split(strip_html(body))
    # Keep the boundaries for lexical linkage and flatten the same words for all excerpts.
    words, anchors = [], {}
    for sentence in sentences:
        chunk = [w[:256] for w in sentence.split()]
        for j, word in enumerate(chunk):
            if YEAR.search(word):
                nearby = key_terms(' '.join(chunk[max(0, j-10):j+11]))
                anchors[len(words) + j] = max(len(nearby & clause) for clause in focus)
        words.extend(chunk)
    last = max(0, len(words) - WINDOW)
    centered = {i: min(max(0, i - WINDOW // 2), last) for i in anchors}
    starts = set(range(0, last+1, WINDOW // 2)) | {last} | set(centered.values())
    windows = {i: ' '.join(words[i:i+WINDOW]) for i in starts}
    lexical = {i: len(key_terms(text) & vocabulary) for i, text in windows.items()}
    plain = max(starts, key=lambda i: (lexical[i], -i))
    dates = {i: plain for i in ARMS}
    linked = [i for i, overlap in anchors.items() if overlap >= 2]
    if linked:
        dates['date_only'] = max(centered.values(), key=lambda i: (lexical[i], -i))
        best = max(linked, key=lambda i: (anchors[i], lexical[centered[i]], -i))
        dates['linked_date'] = centered[best]
    return dates, {arm: windows[i] for arm, i in dates.items()}


def write_gz(path, report):
    with gzip.open(path, 'xt', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False, separators=(',', ':'))


def score(args, groups):
    items = {arm: [] for arm in ARMS}
    rows = []
    for domain, queries in groups.items():
        wanted = {doc for row in queries for _, doc, _ in row['pool']}
        source = read_documents(args.data_root / domain / 'documents.jsonl', wanted)
        for row in queries:
            query = row['query']
            vocabulary = key_terms(query)
            offsets, docs, dense = [], [], []
            for _, doc, value in row['pool']:
                starts, texts = excerpts(source[doc], vocabulary, row['focus'])
                if len({len(text.split()) for text in texts.values()}) != 1:
                    raise ValueError('Expected equal word budgets')
                offsets.append([starts[arm] for arm in ARMS])
                docs.append(doc)
                dense.append(value)
                for arm in ARMS:
                    items[arm].append((query, texts[arm]))
            rows.append({'domain': domain, 'qid': row['qid'], 'docs': docs, 'dense': dense,
                         'excerpt_starts': offsets})
        print(f'{domain}: {len(queries)} questions prepared', flush=True)
    fingerprint = hashlib.sha256(json.dumps(items, ensure_ascii=False).encode('utf-8')).hexdigest()
    unique = list(dict.fromkeys(pair for prompt_list in items.values() for pair in prompt_list))
    print(f'{len(unique)} distinct prompts out of {sum(map(len, items.values()))}', flush=True)
    raw, seconds = score_qwen_variants(args.model, {'matched': unique}, {'matched': INSTRUCT}, 4,
                                      device='cuda', precision='bfloat16')['matched']
    lookup = dict(zip(unique, raw, strict=True))
    for i, row in enumerate(rows):
        row['scores'] = {arm: [lookup[pair] for pair in items[arm][i*POOL:(i+1)*POOL]] for arm in ARMS}
    write_gz(args.output, {'model': args.model.name, 'arms': ARMS, 'device': 'cuda',
                           'population': 'non_history_panel_transfer' if args.transfer else 'outside_panels',
                           'precision': 'bfloat16', 'instruction': INSTRUCT, 'prompt_input_hash': fingerprint,
                           'unique_prompts': len(unique), 'inference_seconds': seconds, 'per_query': rows})


def evaluate(args):
    with gzip.open(args.scores, 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    rows = data['per_query']
    domains = {r['domain'] for r in rows}
    gold = {d: read_qrels(args.data_root / d / 'qrels_train.txt') for d in domains}
    for row in rows:
        relevance = gold[row['domain']].get(row['qid'], set())
        docs = row['docs']
        metrics = {'dense': ndcg10(docs, relevance)}
        baseline = z(row['dense'])
        for arm in ARMS:
            raw = row['scores'][arm]
            if len(raw) != POOL or not all(isfinite(x) for x in raw):
                raise ValueError(f'Incomplete scores for {row["qid"]}')
            for values, label in ((raw, arm), (baseline + z(raw), f'fused_{arm}')):
                order = sorted(range(POOL), key=lambda i: -values[i])
                metrics[label] = ndcg10([docs[i] for i in order], relevance)
        row['ndcg'] = metrics
    measures = {}
    for arm in rows[0]['ndcg']:
        per_domain = {d: mean(r['ndcg'][arm] for r in rows if r['domain'] == d) for d in sorted(domains)}
        measures[arm] = {'macro': mean(per_domain.values()), 'domains': per_domain}
    contrasts = {}
    for after, before in (('linked_date', 'lexical'), ('date_only', 'lexical'),
                          ('linked_date', 'date_only'), ('fused_linked_date', 'fused_lexical'),
                          ('fused_date_only', 'fused_lexical'), ('fused_linked_date', 'fused_date_only')):
        result = paired_delta([(r['domain'], r['ndcg'][after] - r['ndcg'][before]) for r in rows])
        by_domain = result['per_domain']
        result['min_leave_domain_out'] = min(mean(v for k, v in by_domain.items() if k != d)
                                             for d in by_domain) if len(by_domain) > 1 else None
        contrasts[f'{after}_vs_{before}'] = result
    summary = {'queries': len(rows), 'domains': dict(Counter(r['domain'] for r in rows)),
               'measurements': measures, 'contrasts': contrasts,
               'changed_excerpts': {arm: sum(offsets[0] != offsets[i] for r in rows for offsets in r['excerpt_starts'])
                                    for i, arm in enumerate(ARMS[1:], 1)},
               'link_changes_vs_date_only': sum(o[1] != o[2] for r in rows for o in r['excerpt_starts'])}
    write_gz(args.output, data | {'evaluation': summary})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('inventory', 'score', 'evaluate'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--exclude-archive', type=Path)
    parser.add_argument('--exclude-source-cases', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--limit', type=int, default=0, help='Optional per-domain smoke-test limit.')
    parser.add_argument('--transfer', action='store_true',
                        help='Use other-domain questions from the preceding panels.')
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('The smoke-test limit must be nonnegative.')
    if args.stage != 'inventory' and (not args.output or args.output.exists()):
        parser.error('Use a new output path when scoring or evaluating.')
    if args.stage == 'evaluate':
        evaluate(args)
        return
    panels, source_cases = comparison_queries(args.exclude_archive, args.exclude_source_cases)
    groups = sample(args.data_root, args.runs, panels, source_cases, args.limit, args.transfer)
    if not groups:
        raise ValueError('No eligible questions')
    print({domain: len(rows) for domain, rows in groups.items()}, flush=True)
    if args.stage == 'score':
        score(args, groups)


if __name__ == '__main__':
    main()
