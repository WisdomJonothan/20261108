"""Test novelty-aware shortlist construction and larger 8B candidate pools."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from benchmark_pairwise_replacement_v2 import passages, strip_html, terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_shortlist_near_duplicates import greedy, shingles, similarities

SIZES = {'dense34': 34, 'novel34_keep12': 34, 'novel34': 34,
         'dense58': 58, 'novel58_keep24': 58, 'dense100': 100}


def novel(prefix, outside, sim, width):
    kept = list(prefix)
    deferred = []
    for i in outside:
        if len(kept) == width:
            break
        if any(sim[i, j] > 0. for j in kept):
            deferred.append(i)
        else:
            kept.append(i)
    return sorted(kept + deferred[:width-len(kept)])


def pools(docs, start, sim):
    core = [i for i, d in enumerate(docs) if d in start]
    out = [i for i, d in enumerate(docs) if d not in start]
    if len(core) != 10 or len(out) != 90:
        raise ValueError('Expected a ten-document start inside a complete dense pool')
    return {
        'dense34': sorted(core + out[:24]),
        'novel34_keep12': novel(core + out[:12], out[12:], sim, 34),
        'novel34': novel(core, out, sim, 34),
        'dense58': sorted(core + out[:48]),
        'novel58_keep24': novel(core + out[:24], out[24:], sim, 58),
        'dense100': list(range(100)),
    }


def select_queries(root, proxy, old8b, dev8b, dev_reference):
    train_data = {}
    for i in range(3):
        fold = f'f{i}'
        for r in load(root / 'results/replacement_v2' / f'candidates_train_{fold}.json')['queries']:
            train_data[r['domain'], r['qid']] = r | {'fold': fold}
    dev_data = {(r['domain'], r['qid']): r | {'fold': 'dev'}
                for r in load(root / 'results/replacement_v2/candidates_dev.json')['queries']}
    historical = index(load(root / 'results/analysis/position_oracle_oof_20261124.json'))
    dev_six = index(load(dev_reference))
    proxy_ids = {(r['domain'], r['qid']) for r in load(proxy)['per_query']}
    training = [(key, train_data[key] | {'sample': 'proxy'}) for key in proxy_ids]
    earlier = {}
    for path in old8b:
        part = index(load(path))
        if not earlier.keys().isdisjoint(part):
            raise ValueError('Overlapping calibration cohorts')
        earlier.update(part)
    if not proxy_ids.isdisjoint(earlier):
        raise ValueError('Training selection leaks into the 65-query 8B comparison')
    heldout = [(k, train_data[k] | {'sample': 'oof', 'anchor_scores': r['scores']})
               for k, r in earlier.items() if historical[k]['policies']['external6']['chosen'] is None]
    official = [(k, dev_data[k] | {'sample': 'dev', 'anchor_scores': r['scores']})
                for k, r in index(load(dev8b)).items() if dev_six[k]['policies']['external6']['chosen'] is None]
    return training + heldout + official


def produce(root, data, selected):
    domains = defaultdict(list)
    for (d, _), row in selected:
        domains[d].append(row)
    results = []
    for domain, rows in sorted(domains.items()):
        dense = {s: read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_{s}.trec') for s in ('train', 'dev')}
        folds = {}
        for fold in sorted({r['fold'] for r in rows}):
            name = (f'sub_sft_e2bc_{domain}_dev.trec' if fold == 'dev'
                    else f'trainfuse_{fold}_{domain}_train.trec')
            folds[fold] = read_trec_run(root / 'runs' / name)
        prepared = []
        for r in sorted(rows, key=lambda r: (r['sample'], r['qid'])):
            q, fold = r['qid'], r['fold']
            label = 'dev' if fold == 'dev' else 'train'
            docs = [d for _, _, d in dense[label][q][:100]]
            base = {d: s for _, s, d in folds[fold][q][:100]}
            if set(docs) != base.keys() or list(base)[:10] != r['frozen_top10']:
                raise ValueError(f'Different archived rankings for {domain}/{q}')
            prepared.append((r, docs, base, label))
        needed = {doc for _, docs, _, _ in prepared for doc in docs}
        corpus = {}
        with (data / domain / 'documents.jsonl').open(encoding='utf-8') as reader:
            for line in reader:
                item = json.loads(line)
                if item['id'] in needed:
                    corpus[item['id']] = item['content']
                    if corpus.keys() == needed:
                        break
        if corpus.keys() != needed:
            raise ValueError(f'Missing documents for {domain}')
        labels = {s: read_qrels(data / domain / f'qrels_{s}.txt') for s in ('train', 'dev')}
        for r, docs, base, label in prepared:
            q = r['qid']
            gold = labels[label][q]
            query = terms(strip_html(r['query']))
            sim = similarities([shingles(passages(corpus[d], query)[0]) for d in docs])
            masks = pools(docs, set(r['frozen_top10']), sim)
            rankings, scores, oracle = {}, {}, {}
            for arm, locations in masks.items():
                chosen = [docs[i] for i in locations]
                if len(chosen) != SIZES[arm]:
                    raise ValueError(f'Wrong pool width for {domain}/{q}/{arm}')
                values = .7 * z([base[d] for d in chosen])
                if 'anchor_scores' in r:
                    judge = r['anchor_scores']
                    if judge.keys() != set(docs):
                        raise ValueError(f'Score archive differs for {domain}/{q}')
                    values += .3 * z([judge[d] for d in chosen])
                ranking = greedy(chosen, values, sim[np.ix_(locations, locations)], 1.)
                rankings[arm] = ranking
                scores[arm] = ndcg10(ranking, gold)
                oracle[arm] = ndcg10(sorted(gold & set(chosen)), gold)
            results.append({'domain': domain, 'qid': q, 'fold': r['fold'], 'sample': r['sample'],
                            'rankings': rankings, 'scores': scores, 'oracle': oracle,
                            'novel_members_changed': {arm: len(set(loc) - set(masks['dense34']))
                                                      for arm, loc in masks.items() if SIZES[arm] == 34}})
        print(domain, len(results), 'queries prepared', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('data-root', 'proxy', 'dev8b', 'dev-reference', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--oof8b', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    rows = select_queries(args.root, args.proxy, args.oof8b, args.dev8b, args.dev_reference)
    results = produce(args.root, args.data_root, rows)
    numbers = {sample: {arm: macro((r['domain'], r['scores'][arm]) for r in results if r['sample'] == sample)
                        for arm in SIZES} for sample in ('proxy', 'oof', 'dev')}
    chosen_equal_budget = max((arm for arm, width in SIZES.items() if width == 34),
                              key=numbers['proxy'].get)
    chosen_any_budget = max(SIZES, key=numbers['proxy'].get)
    for sample, values in numbers.items():
        print(sample, values, flush=True)
    print('Proxy choices', chosen_equal_budget, chosen_any_budget, flush=True)
    report = {'scores_conditioned_on_abstention': numbers,
              'selected_equal_budget': chosen_equal_budget, 'selected_any_budget': chosen_any_budget,
              'pool_sizes': SIZES, 'per_query': results}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, separators=(',', ':'))
        f.write('\n')


if __name__ == '__main__':
    main()
