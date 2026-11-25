"""Compare same-budget evidence shortlists and a teacher-score student."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean

import numpy as np

from audit_anchor_dev import macro
from benchmark_pairwise_replacement_v2 import passages, strip_html, terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import ordered_scores, paired_delta, z
from replay_anchor_fallback import index

ARMS = ('dense', 'lexical', 'complement', 'student', 'teacher')


def inputs(raw_paths, frozen_path, prior, split):
    available = index(prior['starts']['abstention'])
    frozen = index(load(frozen_path))
    records = []
    for cohort, path in enumerate(raw_paths):
        for key, item in index(load(path)).items():
            policy = available[key]
            if not policy['active']:
                continue
            candidates = frozen[key]['frozen_scores']
            docs = [d for d, _ in candidates]
            start = policy['rankings']['start']
            if (len(set(docs)) != 100 or set(docs) != item['scores'].keys() or
                    set(start) != set(item['arms']['base']['ranking'])):
                raise ValueError(f'Score or abstention inputs disagree for {key}')
            records.append({'domain': key[0], 'qid': key[1], 'cohort': cohort,
                            'docs': docs, 'base': candidates, 'start': start, 'anchor': item['scores']})
    return records


def query_text(root, split):
    source = root / 'results/replacement_v2'
    paths = [source / 'candidates_dev.json'] if split == 'dev' else [
        source / f'candidates_train_f{i}.json' for i in range(3)
    ]
    return {(r['domain'], r['qid']): strip_html(r['query']) for path in paths
            for r in load(path)['queries']}


def features(rows, data, texts, split):
    for domain in sorted({r['domain'] for r in rows}):
        group = [r for r in rows if r['domain'] == domain]
        needed = set().union(*(set(r['docs']) for r in group))
        corpus = {}
        with (data / domain / 'documents.jsonl').open(encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line)
                if doc['id'] in needed:
                    corpus[doc['id']] = doc['content']
                    if len(corpus) == len(needed):
                        break
        if corpus.keys() != needed:
            raise ValueError(f'No content for some candidates in {domain}')
        for r in group:
            query = texts[domain, r['qid']]
            vocabulary = terms(query)
            clauses = [terms(part) for part in re.split(r'(?<=[.!?])\s+', query)]
            clauses = [part for part in clauses if len(part) >= 3] or [vocabulary]
            hits = [terms(passages(corpus[d], vocabulary)[0]) & vocabulary for d in r['docs']]
            frequency = Counter(word for h in hits for word in h)
            weights = {word: 1 + np.log(101 / (1 + frequency[word])) for word in sorted(vocabulary)}

            def coverage(hit, words):
                if not words:
                    return 0.
                return sum(weights[w] for w in sorted(hit & words)) / sum(weights[w] for w in sorted(words))

            lex = np.array([coverage(hit, vocabulary) for hit in hits])
            clause = np.array([[coverage(hit, words) for words in clauses] for hit in hits])
            frozen = z([value for _, value in r['base']])
            dense = z([1 / (61 + i) for i in range(100)])
            target = z([r['anchor'][d] for d in r['docs']])
            r.update(lex=lex, clause=clause, frozen=frozen, target=target,
                     features=np.column_stack([frozen, dense, z(lex), z(clause.max(axis=1)), z(clause.mean(axis=1))]))
        print(split, domain, len(group), 'queries prepared', flush=True)


def fit(rows):
    domains = Counter(r['domain'] for r in rows)
    x = np.concatenate([r['features'] for r in rows])
    y = np.concatenate([r['target'] for r in rows])
    w = np.concatenate([np.full(100, 1. / (100 * len(domains) * domains[r['domain']])) for r in rows])
    return np.linalg.solve(x.T @ (w[:, None] * x) + .1 * np.eye(x.shape[1]), x.T @ (w * y))


def select(r, model):
    known = {i for i, doc in enumerate(r['docs']) if doc in r['start']}
    outsiders = [i for i in range(100) if i not in known]
    first = outsiders[:12]
    rest = outsiders[12:]
    baseline = known | set(outsiders[:24])
    core = known | set(first)
    lexical = sorted(rest, key=lambda i: (-r['lex'][i], i))[:12]
    current = r['clause'][sorted(core)].max(axis=0)
    complement = []
    for _ in range(12):
        next_doc = max((i for i in rest if i not in complement), key=lambda i: (
            np.maximum(r['clause'][i] - current, 0.).sum(), r['lex'][i], -i
        ))
        current = np.maximum(current, r['clause'][next_doc])
        complement.append(next_doc)
    selected = {'dense': baseline, 'lexical': core | set(lexical),
                'complement': core | set(complement)}
    for name, prediction in (('student', r['features'] @ model), ('teacher', r['target'])):
        estimate = .7 * r['frozen'] + .3 * prediction
        ranking = sorted(rest, key=lambda i: (-estimate[i], i))
        selected[name] = core | set(ranking[:12])
    return selected


def replay(rows, model, data, official):
    labels = {d: read_qrels(data / d / ('qrels_dev.txt' if official else 'qrels_train.txt'))
              for d in {r['domain'] for r in rows}}
    result = []
    for r in rows:
        domains, qid = r['domain'], r['qid']
        gold = labels[domains][qid]
        masks = select(r, model)
        item = {'domain': domains, 'qid': qid, 'cohort': r['cohort'],
                'rank': {}, 'score': {}, 'oracle': {}, 'changed_candidates': {}}
        for arm, mask in masks.items():
            if len(mask) != 34:
                raise ValueError(f'Incorrect candidate count for {qid}')
            pool = [entry for i, entry in enumerate(r['base']) if i in mask]
            ranking = ordered_scores(pool, r['anchor'])[:10]
            item['rank'][arm] = ranking
            item['score'][arm] = ndcg10(ranking, gold)
            item['oracle'][arm] = ndcg10(sorted({d for d, _ in pool} & gold), gold)
            item['changed_candidates'][arm] = len(mask - masks['dense'])
        result.append(item)
    return result


def summarize(rows, archived):
    original = {(r['domain'], r['qid']): r for r in rows}
    scope = {(r['domain'], r['qid']): r for r in archived}
    reference = macro((r['domain'], r['scores']['dense_24'] if r['active'] else r['accepted_score']) for r in archived)
    out = {}
    for arm in ARMS:
        pairs, bounds = [], []
        for (d, qid), previous in scope.items():
            if previous['active']:
                r = original[d, qid]
                if abs(r['score']['dense'] - previous['scores']['dense_24']) > 1e-12:
                    raise ValueError(f'Fixed shortlist differs for {d}/{qid}')
                pairs.append((d, r['score'][arm] - r['score']['dense']))
                bounds.append((d, r['oracle'][arm] - r['oracle']['dense']))
            else:
                pairs.append((d, 0.))
                bounds.append((d, 0.))
        paired = paired_delta(pairs)
        out[arm] = {'ndcg': reference + paired['point'], 'change': paired,
                    'oracle_change': paired_delta(bounds),
                    'mean_candidate_changes': mean(r['changed_candidates'][arm] for r in rows)}
    return out


def normalization(rows, data, split):
    labels = {d: read_qrels(data / d / f'qrels_{split}.txt') for d in {r['domain'] for r in rows}}
    result = []
    for r in rows:
        d, q = r['domain'], r['qid']
        gold = labels[d][q]
        known = {i for i, doc in enumerate(r['docs']) if doc in r['start']}
        outsiders = [i for i in range(100) if i not in known]
        core = sorted(known | set(outsiders[:12]))
        raw_base = np.array([value for _, value in r['base']])
        raw_judge = np.array([r['anchor'][doc] for doc in r['docs']])
        frozen_full, judge_full = z(raw_base), z(raw_judge)
        estimate = .7 * frozen_full + .3 * judge_full
        teacher = sorted(outsiders[12:], key=lambda i: (-estimate[i], i))[:12]
        pools = {'dense': sorted(known | set(outsiders[:24])), 'teacher': sorted(set(core) | set(teacher))}
        scored, ranks = {}, {}
        for name, indices in pools.items():
            frozen, judge = raw_base[indices], raw_judge[indices]
            frozen_core = (frozen - raw_base[core].mean()) / max(raw_base[core].std(), 1e-8)
            judge_core = (judge - raw_judge[core].mean()) / max(raw_judge[core].std(), 1e-8)
            settings = {
                'pool_pool': .7 * z(frozen) + .3 * z(judge),
                'core_core': .7 * frozen_core + .3 * judge_core,
                'full_core': .7 * frozen_full[indices] + .3 * judge_core,
                'full_full': estimate[indices],
            }
            scored[name], ranks[name] = {}, {}
            for scope, values in settings.items():
                order = [r['docs'][indices[i]] for i in np.argsort(-values, kind='stable')[:10]]
                scored[name][scope] = ndcg10(order, gold)
                ranks[name][scope] = order
        result.append({'domain': d, 'qid': q, 'scores': scored, 'rankings': ranks})
    return result


def normalization_report(train, dev, data, archive, protected):
    report = {}
    for name, rows, split in (('oof', train, 'train'), ('dev', dev, 'dev')):
        active = normalization(rows, data, split)
        indexed = {(r['domain'], r['qid']): r for r in active}
        baseline = protected[name]['full_policy']['1']['ndcg']
        official = archive[f'{name}_per_query']
        for r in official:
            if r['active'] and abs(indexed[r['domain'], r['qid']]['scores']['dense']['pool_pool']
                                   - r['scores']['dense_24']) > 1e-12:
                raise ValueError(f'Unexpected fixed normalization for {r["qid"]}')
        effects = {}
        for scope in ('pool_pool', 'core_core', 'full_core', 'full_full'):
            diffs = {(r['domain'], r['qid']): {
                source: r['scores'][source][scope] - r['scores']['dense']['pool_pool']
                for source in ('dense', 'teacher')
            } for r in active}
            scores = {}
            for source in ('dense', 'teacher'):
                paired = paired_delta([(r['domain'], diffs.get((r['domain'], r['qid']), {}).get(source, 0.))
                                       for r in official])
                scores[source] = {'ndcg': baseline + paired['point'], 'versus_original': paired}
            scores['teacher_minus_dense'] = paired_delta([
                (r['domain'], diffs[(r['domain'], r['qid'])]['teacher'] -
                 diffs[(r['domain'], r['qid'])]['dense'] if r['active'] else 0.)
                for r in official
            ])
            effects[scope] = scores
            print(name, scope, scores['dense']['ndcg'], scores['teacher']['ndcg'],
                  scores['teacher_minus_dense']['ci95'], flush=True)
        report[name] = {'effects': effects, 'per_query': active}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--budgeted', type=Path, required=True)
    parser.add_argument('--protected', type=Path, required=True)
    parser.add_argument('--oof', type=Path, nargs=2, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    for name in ('oof-scope', 'dev-scope', 'oof-frozen', 'dev-frozen'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--diagnose-normalization', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive, protected = load(args.budgeted), load(args.protected)
    train = inputs(args.oof, args.oof_frozen, load(args.oof_scope), 'train')
    dev = inputs([args.dev], args.dev_frozen, load(args.dev_scope), 'dev')
    if args.diagnose_normalization:
        report = normalization_report(train, dev, args.data_root, archive, protected)
        with args.output.open('x', encoding='utf-8') as file:
            json.dump(report, file, separators=(',', ':'))
            file.write('\n')
        return
    for rows, split in ((train, 'train'), (dev, 'dev')):
        features(rows, args.data_root, query_text(args.root, split), split)
    tested = []
    for i in (0, 1):
        model = fit([r for r in train if r['cohort'] != i])
        tested.extend(replay([r for r in train if r['cohort'] == i], model, args.data_root, False))
    final = fit(train)
    official = replay(dev, final, args.data_root, True)
    report = {'student_coefficients': final.tolist(), 'oof_per_query': tested, 'dev_per_query': official}
    for name, rows in (('oof', tested), ('dev', official)):
        choices = index(protected[name])
        archived = archive[f'{name}_per_query']
        for r in archived:
            r['accepted_score'] = 0. if r['active'] else choices[r['domain'], r['qid']]['scores']['1']
        report[name] = summarize(rows, archived)
        for arm, item in report[name].items():
            print(name, arm, item['ndcg'], item['change']['ci95'],
                  item['oracle_change']['point'], item['mean_candidate_changes'], flush=True)
    report['oof_selected_candidate_policy'] = max(ARMS[:-1], key=lambda arm: report['oof'][arm]['ndcg'])
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
