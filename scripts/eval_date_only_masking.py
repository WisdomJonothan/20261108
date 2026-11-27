"""Test whether masking suspected distractor years helps fixed evidence excerpts."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import strip_html
from coverage_cf_bound import ndcg10, read_qrels
from eval_event_date_windows import SENTENCE, TIME, YEAR, excerpts, key_terms, write_gz
from eval_temporal_query_contrast import score_qwen_variants
from replay_anchor_cap import z
from replay_anchor_edit_scope import paired_delta
from run_qwen3_rerank import INSTRUCT


ARMS = ('original', 'mask_other', 'mask_linked')
ISOLATED_YEAR = re.compile(r'(?<![-–—/])\b(?:1[0-9]{3}|20[0-2][0-9])\b(?![-–—/])')
PLACEHOLDER = '[YEAR]'


def choose_edits(excerpt, focus, protected):
    repeated = Counter(YEAR.findall(excerpt))
    linked, other = [], []
    offset = 0
    for sentence in SENTENCE.split(excerpt):
        # SENTENCE drops separators, so recover each sentence's original offset.
        offset = excerpt.index(sentence, offset)
        edge = offset + len(sentence)
        sentence_overlap = max(len(key_terms(sentence) & clause) for clause in focus)
        for match in ISOLATED_YEAR.finditer(sentence):
            year = match.group()
            if year in protected or repeated[year] > 1:
                continue
            before = sentence[:match.start()].split()[-10:]
            after = sentence[match.end():].split()[:10]
            nearby = key_terms(' '.join(before + after))
            overlap = max(len(nearby & clause) for clause in focus)
            start = offset + match.start()
            if overlap >= 2:
                linked.append((overlap, start, offset))
            elif sentence_overlap == 0:
                other.append((start, offset))
        offset = edge
    candidates = [(overlap, abs(lpos-opos), lpos, opos)
                  for overlap, lpos, ls in linked for opos, os in other if ls != os]
    if not candidates:
        return None
    _, _, lpos, opos = min(candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))
    return {'mask_other': opos, 'mask_linked': lpos}


def replace_year(text, start):
    if not ISOLATED_YEAR.match(text, start):
        raise ValueError('The selected substring must be an isolated year')
    return text[:start] + PLACEHOLDER + text[start+4:]


def read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def prepare(archives, data_root):
    groups = defaultdict(list)
    seen = set()
    for path in archives:
        report = read_gz(path)
        for row in report['per_query']:
            key = (row['domain'], row['qid'])
            if key in seen:
                raise ValueError(f'Duplicated question {key}')
            seen.add(key)
            groups[row['domain']].append((path.name, row))
    prompts = {}
    rows = []
    for domain, group in sorted(groups.items()):
        directory = data_root / domain
        qids = {row['qid'] for _, row in group}
        with (directory / 'examples_train.jsonl').open(encoding='utf-8') as stream:
            queries = {row['id']: strip_html(row['query']) for line in stream
                       for row in (json.loads(line),) if row['id'] in qids}
        documents = read_documents(directory / 'documents.jsonl',
                                   {doc for _, row in group for doc in row['docs']})
        edits = 0
        for source, row in group:
            qid = row['qid']
            text = queries[qid]
            query = ' '.join(text.split())
            focus = [key_terms(s) for s in SENTENCE.split(text) if TIME.search(s)]
            focus = [clause for clause in focus if len(clause) >= 2]
            if not focus:
                raise ValueError(f'Expected original dated query focus for {qid}')
            vocabulary, protected = key_terms(query), set(YEAR.findall(query))
            changes = {}
            for i, doc in enumerate(row['docs']):
                starts, variants = excerpts(documents[doc], vocabulary, focus)
                if starts['lexical'] != row['excerpt_starts'][i][0]:
                    raise ValueError(f'Original evidence offset changed for {qid}/{doc}')
                passage = variants['lexical']
                edit = choose_edits(passage, focus, protected)
                if edit is None:
                    continue
                changes[str(i)] = edit | {'length': len(passage), 'window_start': starts['lexical'],
                                          'excerpt_sha256': hashlib.sha256(passage.encode()).hexdigest()}
                versions = {'original': passage} | {arm: replace_year(passage, start) for arm, start in edit.items()}
                prompts[domain, qid, i] = [(query, versions[arm]) for arm in ARMS]
            rows.append({k: row[k] for k in ('domain', 'qid', 'docs', 'dense')} |
                        {'source': source, 'archived_lexical': row['scores']['lexical'],
                         'archived_lexical_fused_ndcg': row['ndcg']['fused_lexical'], 'edits': changes})
            edits += len(changes)
        print(f'{domain}: {edits} editable excerpts over {len(group)} questions', flush=True)
    return rows, prompts


def score(args):
    rows, prompts = prepare(args.archives, args.data_root)
    unique = list(dict.fromkeys(pair for versions in prompts.values() for pair in versions))
    print(f'{len(prompts)} changed excerpts, {len(unique)} distinct rescore prompts', flush=True)
    if unique:
        raw, seconds = score_qwen_variants(args.model, {'matched': unique}, {'matched': INSTRUCT}, 4,
                                          device='cuda', precision='bfloat16')['matched']
        lookup = dict(zip(unique, raw, strict=True))
    else:
        lookup, seconds = {}, 0.0
    for row in rows:
        row['scores'] = {arm: row['archived_lexical'].copy() for arm in ARMS}
        for i in map(int, row['edits']):
            versions = prompts[row['domain'], row['qid'], i]
            for arm, pair in zip(ARMS, versions, strict=True):
                row['scores'][arm][i] = lookup[pair]
    write_gz(args.output, {'model': args.model.name, 'instruction': INSTRUCT, 'precision': 'bfloat16',
                           'device': 'cuda', 'arms': ARMS, 'unique_prompts': len(unique),
                           'prompt_hash': hashlib.sha256(json.dumps(unique, ensure_ascii=False).encode()).hexdigest(),
                           'inference_seconds': seconds, 'per_query': rows})


def summarize(rows):
    domains = dict(Counter(r['domain'] for r in rows))
    metrics = {}
    for name in rows[0]['ndcg']:
        means = {d: mean(r['ndcg'][name] for r in rows if r['domain'] == d) for d in domains}
        metrics[name] = {'macro': mean(means.values()), 'per_domain': means}
    contrasts = {}
    for prefix in ('', 'fused_'):
        for reference in ('original', 'mask_linked'):
            a, b = prefix+'mask_other', prefix+reference
            delta = paired_delta([(r['domain'], r['ndcg'][a] - r['ndcg'][b]) for r in rows])
            contrasts[f'{a}_vs_{b}'] = delta
    edited = [(r, int(i)) for r in rows for i in r['edits']]
    gold_edits = sum(i in r['edited_positive_indices'] for r,i in edited)
    score_drift = [abs(r['archived_lexical'][i] - r['scores']['original'][i]) for r,i in edited]
    return {'queries': len(rows), 'domains': domains, 'editable_queries': sum(bool(r['edits']) for r in rows),
            'candidates': sum(len(r['docs']) for r in rows), 'editable_excerpts': len(edited),
            'edited_judged_positive_excerpts': gold_edits,
            'baseline_drift': {'mean_absolute_raw': mean(score_drift) if score_drift else 0,
                               'max_absolute_raw': max(score_drift, default=0)},
            'ndcg': metrics, 'contrasts': contrasts}


def evaluate(args):
    data = read_gz(args.scores)
    rows = data['per_query']
    gold = {d: read_qrels(args.data_root / d / 'qrels_train.txt') for d in {r['domain'] for r in rows}}
    for row in rows:
        relevance = gold[row['domain']].get(row['qid'], set())
        docs = row['docs']
        row['edited_positive_indices'] = [i for i in map(int, row['edits']) if docs[i] in relevance]
        dense = z(row['dense'])
        row['ndcg'] = {'archived_fused_lexical': row['archived_lexical_fused_ndcg']}
        for arm in ARMS:
            raw = row['scores'][arm]
            if len(raw) != len(docs) or not all(isfinite(value) for value in raw):
                raise ValueError(f'Incomplete scores for {row["qid"]}')
            for values, label in ((raw, arm), (dense + z(raw), f'fused_{arm}')):
                ranked = sorted(range(len(docs)), key=lambda i: -values[i])
                row['ndcg'][label] = ndcg10([docs[i] for i in ranked], relevance)
    cohorts = sorted({r['source'] for r in rows})
    results = {cohort: summarize([r for r in rows if r['source'] == cohort]) for cohort in cohorts}
    write_gz(args.output, data | {'evaluation': results})
    for cohort, result in results.items():
        print(cohort, {k:v for k,v in result.items() if k not in ('contrasts', 'ndcg')}, flush=True)
        print({k: round(v['macro'], 6) for k,v in result['ndcg'].items()}, flush=True)
        for k, v in result['contrasts'].items():
            print(k, {key: v[key] for key in ('point', 'ci95', 'improved_queries', 'declined_queries')}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('score', 'evaluate'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--archives', type=Path, nargs='+')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    if args.stage == 'score':
        if args.archives is None or args.model is None:
            parser.error('Scoring requires --archives and --model.')
        score(args)
    else:
        if args.scores is None:
            parser.error('Evaluation requires --scores.')
        evaluate(args)


if __name__ == '__main__':
    main()
