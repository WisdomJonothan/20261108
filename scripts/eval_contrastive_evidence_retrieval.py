"""Retrieve candidates by contrasting a missing and a represented query fragment."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np

from audit_wide_vote_contributions import member
from benchmark_pairwise_replacement_v2 import strip_html, terms
from build_temporal_query_views import PREFIX, fragments
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from eval_event_date_windows import write_gz
from fit_dense_query_residual import domain_index
from lib_reteco import load_examples, read_trec_run
from replay_anchor_edit_scope import paired_delta


ARMS = ('original', 'gap', 'contrast', 'reversal')
PANELS = ('held', 'confirm')


def identify_fragments(text):
    sections = list(dict.fromkeys(part for part in fragments(strip_html(text))
                                  if part.endswith('?') and len(terms(part)) >= 3))[:6]
    return [sections[0], *[sections[0] + ' ' + clause for clause in sections[1:]]] if sections else []


def normalized(values):
    return values / np.linalg.norm(values)


def search_vectors(query, facets, anchors):
    fallback = np.stack([query]*len(ARMS))
    if len(facets) < 2:
        return fallback, None
    covered = (facets @ anchors.T).max(axis=1)
    low, high = int(covered.argmin()), int(covered.argmax())
    if low == high or float(covered[high]-covered[low]) < 1e-7:
        return fallback, None
    difference = facets[low] - facets[high]
    directions = (facets[low], difference, -difference)
    tangent = [d - np.dot(query, d)*query for d in directions]
    if min(np.linalg.norm(d) for d in tangent) < 1e-7:
        return fallback, None
    forms = np.stack([query, *[normalized(query + .3*normalized(d)) for d in tangent]])
    return forms, {'gap': low, 'covered': high, 'coverage': covered.tolist()}


def archived_rows(root, panels):
    groups = defaultdict(list)
    sizes = {'held': 186, 'confirm': 155, 'reserve': 191}
    fields = {'held': 'wide-vote-held-calibrated.json', 'confirm': 'wide-vote-confirm-result.json',
              'reserve': 'wide-vote-reserve-result.json'}
    seen = set()
    for panel in panels:
        archive = root/f'wide-vote-{panel}-20261126.tar.gz'
        scopes = member(archive, f'wide-vote-{panel}-scopes.json')['queries']
        rankings = {(r['domain'], r['qid']): r['rankings']['current']
                    for r in member(archive, fields[panel])['per_query']}
        if len(scopes) != sizes[panel] or len(rankings) != sizes[panel]:
            raise ValueError(f'Unexpected panel {panel}')
        for row in scopes:
            d, qid = row['domain'], row['qid']
            key = (d, qid)
            if key in seen or row['split'] != 'train':
                raise ValueError('Expected disjoint training panels')
            seen.add(key)
            pool = row['original']
            if len(pool) != 60 or len(set(pool)) != 60:
                raise ValueError(f'Expected original 60 IDs for {key}')
            groups[d].append({'domain': d, 'qid': qid, 'panel': panel, 'pool': pool,
                              'current': rankings[key]})
    return groups


def interleave(first, second):
    return list(dict.fromkeys(doc for pair in zip(first, second, strict=True) for doc in pair))[:4]


def prepare(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from run_dense_generic import encode

    torch.set_num_threads(3)
    groups = archived_rows(args.archives, ('reserve',) if args.reserve else PANELS)
    total = sum(map(len, groups.values()))
    model = AutoModel.from_pretrained(args.encoder, local_files_only=True).to('cuda', dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)
    output = []
    for domain, rows in sorted(groups.items()):
        path = args.data_root/domain
        questions = dict(zip(*load_examples(path, 'train'), strict=True))
        ids, memmap = domain_index(path, args.embeddings)
        index = {doc: i for i, doc in enumerate(ids)}
        run = read_trec_run(args.runs/f'dense_mxbai_{domain}_train.trec')
        views = [[questions[r['qid']], *identify_fragments(questions[r['qid']])] for r in rows]
        encoded = encode([PREFIX+t for group in views for t in group], model, tokenizer, 'cuda',
                         16, 512, 'mean', f'{domain} full and literal requests')
        cursor = 0
        vectors = []
        for row, texts in zip(rows, views, strict=True):
            query, *facets = encoded[cursor:cursor+len(texts)]
            cursor += len(texts)
            anchor = memmap[[index[d] for d in row['current']]]
            forms, match = search_vectors(query, np.array(facets), anchor)
            vectors.append(forms)
            row['contrast'] = match
            row['fragments'] = len(facets)
        corpus = torch.tensor(memmap, device='cuda', dtype=torch.float32)
        for start in range(0, len(rows), 16):
            group = rows[start:start+16]
            q = torch.tensor(np.concatenate(vectors[start:start+16]), device='cuda', dtype=torch.float32)
            sims = (q @ corpus.T).reshape(len(group), len(ARMS), len(ids))
            official = sims[:, 0].topk(100, dim=1).indices.tolist()
            forbidden = torch.tensor([[index[d] for d in row['pool']] for row in group], device='cuda')
            sims.scatter_(2, forbidden[:, None, :].expand(-1,len(ARMS),-1), -torch.inf)
            indices = sims.topk(4, dim=2).indices.tolist()
            for row, original, proposals in zip(group, official, indices, strict=True):
                previous = {d for _, _, d in run[row['qid']][:100]}
                row['original_replay_overlap'] = len(previous & {ids[i] for i in original}) / 100
                offers = {arm: [ids[i] for i in offer] for arm, offer in zip(ARMS, proposals, strict=True)}
                if args.reserve:
                    offers['two_sided'] = interleave(offers['contrast'], offers['reversal'])
                    offers['original_gap_mix'] = interleave(offers['original'], offers['gap'])
                row['offers'] = offers
            del sims
        del corpus
        output.extend(rows)
        print(f'{domain}: {len(output)} / {total} retrieved', flush=True)
    overlap = mean(r['original_replay_overlap'] for r in output)
    if overlap < .9:
        raise ValueError(f'Historical full-query baseline mismatch ({overlap:.4f})')
    write_gz(args.output, {'encoder': args.encoder.name, 'original_replay_overlap': overlap, 'per_query': output})
    print(hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)


def summarize(rows):
    arms = {}
    for arm in rows[0]['offers']:
        arms[arm] = {'new_positives': sum(r['outcomes'][arm]['positives'] for r in rows),
                     'queries_with_positive': sum(r['outcomes'][arm]['positives'] > 0 for r in rows),
                     'new_steps': sum(r['outcomes'][arm]['new_steps'] for r in rows),
                     'full_oracle': paired_delta([(r['domain'], r['outcomes'][arm]['full_oracle']) for r in rows])['point'],
                     'tail_oracle': paired_delta([(r['domain'], r['outcomes'][arm]['tail_oracle']) for r in rows])['point']}
    compare = {}
    for control in ('original', 'gap', 'reversal'):
        compare[control] = {metric: paired_delta([
            (r['domain'], r['outcomes']['contrast'][metric] - r['outcomes'][control][metric]) for r in rows])
            for metric in ('full_oracle', 'tail_oracle')}
    return {'queries': len(rows), 'domains': dict(Counter(r['domain'] for r in rows)),
            'active': sum(r['contrast'] is not None for r in rows), 'annotated': sum(r['annotated'] for r in rows),
            'arms': arms, 'contrast_comparisons': compare}


def score(args):
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if digest != args.fingerprint:
        raise ValueError('Frozen candidate manifest does not match')
    with gzip.open(args.manifest, 'rt', encoding='utf-8') as stream:
        report = json.load(stream)
    rows = report['per_query']
    domains = {r['domain'] for r in rows}
    qrels = {d: read_qrels(args.data_root/d/'qrels_train.txt') for d in domains}
    steps = {d: read_steps(args.data_root/d/'qrels_steps_train.txt') for d in domains}
    for row in rows:
        domain, qid = row['domain'], row['qid']
        positive = qrels[domain][qid]
        current = set(row['current'])
        missing = [docs for docs in steps[domain].get(qid, {}).values() if docs.isdisjoint(current)]
        row['annotated'] = qid in steps[domain]
        row['outcomes'] = {}
        for arm, offers in row['offers'].items():
            if len(set(offers)) != 4 or not set(offers).isdisjoint(row['pool']):
                raise ValueError(f'Candidate budget differs for {domain}/{qid}')
            present = set(offers) & positive
            ceiling = set(row['pool']) & positive
            swap = row['current']
            if present and swap[-1] not in positive:
                swap = swap[:9] + [min(present)]
            row['outcomes'][arm] = {
                'positives': len(present), 'new_steps': sum(not docs.isdisjoint(offers) for docs in missing),
                'full_oracle': ndcg10(sorted(ceiling | present), positive),
                'tail_oracle': ndcg10(swap, positive),
            }
    panels = {r['panel'] for r in rows}
    if panels == {'reserve'}:
        summary = summarize(rows)
        matched = {}
        for control in ('original', 'original_gap_mix'):
            matched[control] = {
                metric: paired_delta([(r['domain'], r['outcomes']['two_sided'][metric]
                                       - r['outcomes'][control][metric]) for r in rows])
                for metric in ('full_oracle', 'tail_oracle')
            }
        summary['two_sided_comparisons'] = matched
        summaries = {'reserve': summary}
        eligible = all(matched[c]['full_oracle']['point'] > 0
                       and summary['arms']['two_sided']['new_steps'] > summary['arms'][c]['new_steps']
                       for c in matched)
    elif panels == set(PANELS):
        summaries = {panel: summarize([r for r in rows if r['panel'] == panel]) for panel in PANELS}
        summaries['pooled'] = summarize(rows)
        eligible = (all(summaries[p]['contrast_comparisons']['gap']['full_oracle']['point'] > 0 for p in PANELS)
                    and summaries['pooled']['arms']['contrast']['new_steps']
                    > summaries['pooled']['arms']['gap']['new_steps'])
    else:
        raise ValueError('Unexpected archived panels')
    write_gz(args.output, report | {'summary': summaries, 'eligible': eligible, 'rank_fingerprint': digest})
    for panel, summary in summaries.items():
        print(panel, summary['active'], '/', summary['queries'], summary['arms'], flush=True)
    print('eligible:', eligible, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='stage', required=True)
    first = subs.add_parser('prepare')
    for name in ('archives', 'embeddings', 'runs', 'encoder'):
        first.add_argument(f'--{name}', type=Path, required=True)
    first.add_argument('--reserve', action='store_true')
    last = subs.add_parser('score')
    last.add_argument('--manifest', type=Path, required=True)
    last.add_argument('--fingerprint', required=True)
    for stage in (first, last):
        for name in ('data-root', 'output'):
            stage.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    (prepare if args.stage == 'prepare' else score)(args)


if __name__ == '__main__':
    main()
