"""Compare fixed-size reranking shortlists on top of an archived RETECO run."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index

SOURCES = ('dense_mxbai', 'dense_e5base', 'dense_bge')
ARMS = ('within100', 'depth300', 'other_retrievers')
WIDTH = 6


def build(root, reference, routes):
    starts = index(load(reference)['starts']['abstention'])
    routing = index(load(routes))
    if starts.keys() != routing.keys() or len(starts) != 519:
        raise ValueError('Rankings and routes must cover the same 519 official queries')
    rows = []
    for domain in sorted({d for d, _ in starts}):
        ranked = [read_trec_run(root / 'runs' / f'{source}_{domain}_dev.trec') for source in SOURCES]
        for d, qid in sorted(starts):
            if d != domain:
                continue
            current = starts[d, qid]['rankings']['unrestricted']
            lists = [[doc for _, _, doc in run[qid][:300]] for run in ranked]
            if len(set(lists[0])) != 300 or not set(current) <= set(lists[0][:100]):
                raise ValueError(f'Invalid dense pool or starting list for {domain}/{qid}')
            scores = defaultdict(float)
            for docs in lists:
                for position, doc in enumerate(dict.fromkeys(docs), 1):
                    scores[doc] += 1 / (60 + position)
            outside = (set(lists[1]) | set(lists[2])) - set(lists[0])
            pools = (set(lists[0][:100]) - set(current), set(lists[0][100:]), outside)
            shortlists = {
                arm: sorted(pool, key=lambda doc: (-scores[doc], doc))[:WIDTH]
                for arm, pool in zip(ARMS, pools)
            }
            if any(len(docs) != WIDTH for docs in shortlists.values()):
                raise ValueError(f'Cannot fill all shortlists for {domain}/{qid}')
            shortlists['dense_first_six'] = lists[0][100:106]
            rows.append({
                'domain': domain, 'qid': qid, 'start': current, 'shortlists': shortlists,
                'rank_disagreement': bool(routing[d, qid]['expanded']['rank_disagreement']),
                'abstention': routing[d, qid]['abstained'],
            })
    return rows


def audit(rows, data, scored=None, boundary_only=False):
    gold = {d: read_qrels(data / d / 'qrels_dev.txt') for d in {r['domain'] for r in rows}}
    results = []
    for row in rows:
        d, q = row['domain'], row['qid']
        relevant = gold[d][q]
        start = row['start']
        baseline = ndcg10(start, relevant)
        shortlists = row['shortlists']
        best = {
            arm: max(baseline, *(ndcg10(start[:9] + [doc], relevant) for doc in docs))
            for arm, docs in shortlists.items()
        }
        item = {
            'domain': d, 'qid': q, 'baseline': baseline, 'oracle_slot10': best,
            'shortlist_relevant': {arm: len(set(docs) & relevant) for arm, docs in shortlists.items()},
            'rank_disagreement': row['rank_disagreement'], 'abstention': row['abstention'],
        }
        if scored is not None:
            scores = scored[d, q]['scores']
            incumbents = start[9:] if boundary_only else start
            expected = set(incumbents).union(*row['shortlists'].values())
            if scores.keys() != expected:
                raise ValueError(f'Unexpected scored documents for {d}/{q}')
            def ordered(docs):
                return sorted(docs, key=lambda doc: -scores[doc])
            rankings = {} if boundary_only else {'same_ten': ordered(start)}
            for arm, docs in row['shortlists'].items():
                incoming = ordered(docs)[0]
                replaced = scores[incoming] > scores[start[9]]
                new = start[:9] + [incoming] if replaced else start
                rankings[arm] = new
                if not boundary_only:
                    rankings[arm + '_sorted'] = ordered(new)
            item['rankings'] = rankings
            item['ndcg'] = {arm: ndcg10(ranking, relevant) for arm, ranking in rankings.items()}
            item['scores'] = scores
        results.append(item)
    return results


def summarize(rows, field):
    arms = rows[0][field]
    baseline = paired_delta([(r['domain'], r['baseline']) for r in rows])['point']
    summary = {'baseline': baseline}
    for arm in arms:
        delta = paired_delta([(r['domain'], r[field][arm] - r['baseline']) for r in rows])
        summary[arm] = {'ndcg': baseline + delta['point'], 'change': delta}
        if field == 'oracle_slot10':
            summary[arm]['queries_with_relevant_shortlist'] = sum(r['shortlist_relevant'][arm] > 0 for r in rows)
        if field == 'ndcg':
            summary[arm]['recovered_zeroes'] = sum(r['baseline'] == 0 < r[field][arm] for r in rows)
            summary[arm]['new_zeroes'] = sum(r[field][arm] == 0 < r['baseline'] for r in rows)
    if field == 'ndcg':
        for route in ('rank_disagreement', 'abstention'):
            change = paired_delta([(r['domain'], (r[field]['depth300'] - r['baseline']) * r[route]) for r in rows])
            summary['depth300_on_' + route] = {'ndcg': baseline + change['point'], 'change': change}
        summary['depth300_route_comparison'] = paired_delta([
            (r['domain'], (r[field]['depth300'] - r['baseline']) *
             (int(r['rank_disagreement']) - int(r['abstention']))) for r in rows
        ])
    return summary


def score(rows, root, model_path, batch_size, checkpoint, boundary_only):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import benchmark_pairwise_replacement_v2 as judge
    from audit_anchor_prompt_outliers import trim_words
    from eval_anchor_reranker import write_checkpoint

    queries = {(r['domain'], r['qid']): r['query']
               for r in load(root / 'results/replacement_v2/candidates_dev.json')['queries']}
    checkpoint.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    labels = [tokenizer.encode(word, add_special_tokens=False) for word in ('yes', 'no')]
    if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
        raise ValueError('Expected distinct one-token yes/no decisions')
    yes, no = (ids[0] for ids in labels)
    out = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for domain in sorted({r['domain'] for r in rows}):
        domain_rows = [r for r in rows if r['domain'] == domain]
        documents = {}
        for row in domain_rows:
            incumbents = row['start'][9:] if boundary_only else row['start']
            documents[row['qid']] = list(dict.fromkeys(incumbents + sum(row['shortlists'].values(), [])))
        needs = set().union(*documents.values())
        corpus = {}
        with (root / 'data/track1_tempo' / domain / 'documents.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                doc = json.loads(line)
                if doc['id'] in needs:
                    corpus[doc['id']] = doc['content']
                    if len(corpus) == len(needs):
                        break
        if corpus.keys() != needs:
            raise ValueError(f'Missing document text in {domain}')
        for r in domain_rows:
            qid = r['qid']
            docs = documents[qid]
            x = {'domain': domain, 'qid': qid, 'model': model_path.name, 'documents': docs,
                 'batch_size': batch_size}
            path = checkpoint / f'{domain}-{qid}.json'
            if path.exists():
                saved = load(path)
                if any(saved[k] != v for k, v in x.items()):
                    raise ValueError(f'Checkpoint inputs differ for {domain}/{qid}')
                out.append(saved)
                continue
            query = judge.strip_html(queries[domain, qid])
            terms = judge.terms(query)
            passages = [trim_words(judge.passages(corpus[doc], terms)[0], 2048, 256) for doc in docs]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': judge.perdoc_prompt(query, passage)}],
                tokenize=False, add_generation_prompt=True) for passage in passages]
            values, tokens = [], 0
            with torch.inference_mode():
                for offset in range(0, len(prompts), batch_size):
                    batch = tokenizer(prompts[offset:offset + batch_size], add_special_tokens=False,
                                      padding=True, return_tensors='pt').to(model.device)
                    logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                    values.extend((logits[:, yes] - logits[:, no]).cpu().tolist())
                    tokens += int(batch['attention_mask'].sum())
            item = x | {'scores': dict(zip(docs, values)), 'tokens': tokens}
            write_checkpoint(path, item)
            out.append(item)
        print(f'{domain}: {len(out)}/{len(rows)} queries in {time.perf_counter() - begin:.1f}s', flush=True)
    stats = {'prompt_count_including_cache': sum(len(r['scores']) for r in out),
             'tokens_including_cache': sum(r['tokens'] for r in out),
             'wall_seconds': time.perf_counter() - begin,
             'gpu_peak_mib': torch.cuda.max_memory_allocated() / 2**20}
    return index({'per_query': out}), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--routes', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--write-manifest', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--boundary-only', action='store_true',
                        help='Score only rank ten and the shortlists for fixed-slot replacement.')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.write_manifest and args.write_manifest.exists()):
        parser.error('Use fresh output paths.')
    if args.manifest:
        rows = load(args.manifest)['queries']
    elif args.reference and args.routes:
        rows = build(args.root, args.reference, args.routes)
    else:
        parser.error('Supply --manifest or a reference and routes.')
    if args.write_manifest:
        with args.write_manifest.open('x', encoding='utf-8') as stream:
            json.dump({'queries': rows}, stream, separators=(',', ':'))
    scored, stats = None, None
    if args.model:
        if not args.checkpoint or args.batch_size < 1:
            parser.error('GPU evaluation requires a checkpoint directory and positive batch size.')
        scored, stats = score(rows, args.root, args.model, args.batch_size, args.checkpoint, args.boundary_only)
    items = audit(rows, args.root / 'data/track1_tempo', scored, args.boundary_only)
    report = {
        'queries': len(items), 'domains': len({r['domain'] for r in items}),
        'shortlist_width': WIDTH, 'boundary_only': args.boundary_only,
        'oracle_slot10': summarize(items, 'oracle_slot10'),
    }
    if scored is not None:
        report |= {'model': args.model.name, 'inference': stats, 'actual': summarize(items, 'ndcg')}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report | {'per_query': items}, stream, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
