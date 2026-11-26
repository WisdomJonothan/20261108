"""Lock external benchmark rank replays before opening relevance judgments."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import log2
from pathlib import Path
from statistics import mean
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from official.ir_metrics import ndcg_at_k
from replay_anchor_edit_scope import paired_delta


REVISION = '288e9897af910f18934adb924034318406ea680e'
ROOT = f'https://raw.githubusercontent.com/aminbigdeli/ADORE/{REVISION}/'
DOMAINS = ('biology', 'earth-science', 'economics', 'psychology', 'robotics',
           'stackoverflow', 'sustainable-living')
BENCHMARKS = {'BRIGHT': DOMAINS, 'BEIR': ('dbpedia-entity', 'fiqa', 'scifact', 'trec-covid', 'trec-news')}
METHODS = ('query2doc_cot', 'query2doc_fs', 'query2doc_zs', 'genqr', 'adore', 'thinkqe')
K = 10


def download(path):
    for delay in (1, 3, None):
        try:
            with urlopen(ROOT + path, timeout=18) as response:
                raw = response.read()
            return raw.decode('utf-8'), sha256(raw).hexdigest()
        except (URLError, TimeoutError) as error:
            if delay is None or (isinstance(error, HTTPError) and 400 <= error.code < 429):
                raise RuntimeError(f'Download failed for {path}') from error
            sleep(delay)


def parse_run(text):
    queries = defaultdict(dict)
    for line in text.splitlines():
        if not line.strip():
            continue
        qid, _, doc, rank, _, _ = line.split()
        if doc in queries[qid]:
            raise ValueError(f'Duplicate document in the run for {qid}')
        queries[qid][doc] = int(rank)
    if not queries or any(sorted(order.values()) != list(range(1, K + 1))
                          for order in queries.values()):
        raise ValueError('Expected consecutive ranks one through ten for each query')
    return dict(queries)


def fuse(views, weights):
    score = defaultdict(float)
    for view, weight in zip(views, weights, strict=True):
        for doc, rank in sorted(view.items(), key=lambda item: item[1]):
            score[doc] += weight / (60 + rank)
    return sorted(score, key=score.__getitem__, reverse=True)[:K]


def full_fusions(views):
    return {'rrf_all': fuse(views, (1, 1, 1, 1, 1, 1)),
            'rrf_family': fuse(views, (1/3, 1/3, 1/3, 1, 1, 1))}


def rank(views):
    current = fuse(views[:4], (1, 1, 1, 1))
    proposer, witness = views[4:]
    entrant = next((doc for doc in sorted(proposer, key=proposer.__getitem__)
                    if doc not in current), None)
    incumbent = current[-1]
    nominated = entrant is not None and proposer[entrant] < proposer.get(incumbent, K + 1)
    entrant_supported = nominated and entrant in witness
    paired = entrant_supported and witness[entrant] < witness.get(incumbent, K + 1)
    rankings = {
        'current': current,
        'entrant_only': current[:-1] + [entrant] if entrant_supported else current,
        'paired': current[:-1] + [entrant] if paired else current,
        **full_fusions(views),
    }
    return rankings, {'proposed': nominated, 'entrant_supported': entrant_supported,
                      'paired': paired}


def prepare(path, benchmark):
    rows, sources = [], {}
    paths = [f'runs/{benchmark}/{domain}/{name}/runs/bm25_run.txt'
             for domain in BENCHMARKS[benchmark] for name in METHODS]
    with ThreadPoolExecutor(max_workers=3) as executor:
        retrieved = dict(zip(paths, executor.map(download, paths), strict=True))
    for offset, domain in enumerate(BENCHMARKS[benchmark]):
        runs = []
        for source in paths[offset*len(METHODS):(offset+1)*len(METHODS)]:
            contents, sources[source] = retrieved[source]
            runs.append(parse_run(contents))
        ids = runs[0].keys()
        if any(run.keys() != ids for run in runs[1:]):
            raise ValueError(f'The six run populations differ in {domain}')
        for qid in sorted(ids):
            views = [run[qid] for run in runs]
            row = {'domain': domain, 'qid': qid}
            if benchmark == 'BRIGHT':
                row['rankings'], row['decisions'] = rank(views)
            else:
                row['rankings'] = full_fusions(views)
            rows.append(row)
        print(domain, len(ids), flush=True)
    data = {'revision': REVISION, 'run_sha256': sources, 'per_query': rows}
    if benchmark != 'BRIGHT':
        data['benchmark'] = benchmark
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, separators=(',', ':'))
    print('Rankings SHA-256', sha256(path.read_bytes()).hexdigest(), flush=True)


def parse_qrels(contents, binary):
    qrels = defaultdict(dict)
    for line in contents.splitlines():
        if not line.strip():
            continue
        qid, _, doc, grade = line.split()
        value = int(grade)
        if value < 0 or (binary and value not in (0, 1)):
            raise ValueError('Unexpected assessment grade')
        qrels[qid][doc] = value
    return dict(qrels)


def graded_ndcg(order, gold):
    discount = [1 / log2(rank + 2) for rank in range(K)]
    ideal = sum(g * w for g, w in zip(sorted(gold.values(), reverse=True), discount))
    return sum(gold.get(doc, 0) * w for doc, w in zip(order, discount)) / ideal if ideal else 0.


def compare(rows, method, reference):
    effect = paired_delta([(r['domain'], r['scores'][method] - r['scores'][reference]) for r in rows])
    values = effect['per_domain']
    effect['minimum_leave_one_domain_out'] = min(
        mean(v for d, v in values.items() if d != excluded) for excluded in values
    )
    return effect


def score(path, fingerprint, output, benchmark):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != fingerprint:
        raise ValueError('Rankings differ from the fingerprint recorded before scoring')
    saved = json.loads(raw)
    if saved['revision'] != REVISION or saved.get('benchmark', 'BRIGHT') != benchmark:
        raise ValueError('Unexpected data version')
    rows = saved['per_query']
    scores, labels = {}, {}
    for row in rows:
        scores.setdefault(row['domain'], {})[row['qid']] = row
    domains = BENCHMARKS[benchmark]
    if tuple(scores) != domains or sum(map(len, scores.values())) != len(rows):
        raise ValueError('The rank file does not cover the fixed collections exactly once')
    for domain in domains:
        contents, labels[domain] = download(f'qrels/{benchmark}/{domain}.qrels')
        gold = parse_qrels(contents, benchmark == 'BRIGHT')
        if gold.keys() != scores[domain].keys():
            raise ValueError(f'Qrels and rankings have different query populations in {domain}')
        for qid, row in scores[domain].items():
            row['scores'] = {arm: graded_ndcg(order, gold[qid]) for arm, order in row['rankings'].items()}
            if benchmark == 'BRIGHT':
                binary = {doc for doc, value in gold[qid].items() if value}
                if any(abs(row['scores'][arm]-ndcg_at_k(order, binary)) > 1e-12
                       for arm, order in row['rankings'].items()):
                    raise ValueError('Binary nDCG failed the existing scorer cross-check')
    comparisons = [('rrf_family', 'rrf_all')]
    if benchmark == 'BRIGHT':
        comparisons = [('entrant_only', 'current'), ('paired', 'current'),
                       ('paired', 'entrant_only'), ('rrf_all', 'current'), *comparisons]
    contrasts = {f'{a}_vs_{b}': compare(rows, a, b) for a, b in comparisons}
    summary = {
        'rank_sha256': fingerprint, 'revision': REVISION, 'qrels_sha256': labels,
        'queries_by_domain': {d: len(scores[d]) for d in domains},
        'scores': {arm: mean(mean(r['scores'][arm] for r in group.values())
                             for group in scores.values()) for arm in rows[0]['scores']},
        'contrasts': contrasts,
    }
    if benchmark == 'BRIGHT':
        relevant = [contrasts['paired_vs_current'], contrasts['paired_vs_entrant_only']]
        summary['decisions'] = {name: sum(r['decisions'][name] for r in rows)
                                for name in ('proposed', 'entrant_supported', 'paired')}
        summary['paired_eligible'] = all(min(e['point'], e['minimum_leave_one_domain_out']) > 0
                                         for e in relevant)
    with output.open('x', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'per_query': rows}, handle, separators=(',', ':'))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'score'):
        mode = modes.add_parser(command)
        mode.add_argument('--rankings', type=Path, required=True)
        mode.add_argument('--benchmark', choices=BENCHMARKS, default='BRIGHT')
        if command == 'score':
            mode.add_argument('--fingerprint', required=True)
            mode.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.rankings, args.benchmark)
    else:
        score(args.rankings, args.fingerprint, args.output, args.benchmark)


if __name__ == '__main__':
    main()
