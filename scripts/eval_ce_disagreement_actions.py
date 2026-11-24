"""Reserve three of six judge calls for distinct candidates favored by CE."""
import argparse
import json
from collections import Counter
from pathlib import Path

from coverage_cf_bound import DOMAINS, ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_position_discount import bootstrap, eligible


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def action_key(action):
    return action['slot'], action['challenger']


def build(root, fold, with_text):
    official = fold == 'dev'
    split = 'dev' if official else 'train'
    suffix = 'dev' if official else f'train_{fold}'
    tag = 'fusion100_full_ce_s17_20261115' if official else f'{fold}_ce'
    directory = root / 'results/replacement_v2'
    manifest = load(directory / f'candidates_{suffix}.json')['queries']
    cache = load(directory / f'expanded3_fixed_{suffix}.json')['budgets']['18']['per_query']
    history = {(r['domain'], r['qid']): r for r in cache}
    assert len(manifest) == len(history)
    output = []
    for domain in DOMAINS:
        data = root / 'data/track1_tempo' / domain
        gold = read_qrels(data / f'qrels_{split}.txt')
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_{split}.trec')
        ce = read_trec_run(root / 'runs' / f'{tag}_{domain}_{split}_ce.trec')
        built = []
        for item in (r for r in manifest if r['domain'] == domain):
            qid, top = item['qid'], item['frozen_top10']
            old = history[domain, qid]
            ordered = sorted(old['actions'], key=lambda a: (a['slot'], -a['prior']))
            baseline = ndcg10(top, gold[qid])
            assert abs(old['baseline'] - baseline) < 1e-10 and len(ordered) >= 6
            ranks = {d: rank for rank, _, d in ce[qid]}
            dense_top20 = {d for _, _, d in dense[qid][:20]}
            used = set(top) | {a['challenger'] for a in ordered[:3]}
            # Hold the head positions available. Among the first three documents
            # use the one that the independent CE ranks lowest.
            slot = max(range(3), key=lambda i: (ranks[top[i]], -i))
            picked = [d for rank, _, d in ce[qid]
                      if rank <= 20 and d not in used and d not in dense_top20
                      and rank < ranks[top[slot]]][:3]
            scored = {action_key(a): a for a in ordered}
            selected = ordered[:3].copy()
            for doc in picked:
                ranking = top.copy()
                ranking[slot] = doc
                proposal = {'slot': slot + 1, 'incumbent': top[slot], 'challenger': doc,
                            'delta': ndcg10(ranking, gold[qid]) - baseline}
                selected.append(scored.get(action_key(proposal), proposal))
            occupied = {action_key(a) for a in selected}
            selected.extend([a for a in ordered if action_key(a) not in occupied][:6 - len(selected)])
            assert len({action_key(a) for a in selected}) == 6
            built.append({'domain': domain, 'qid': qid, 'query': item['query'],
                          'baseline': baseline, 'control': ordered[:6], 'treatment': selected,
                          'new_disagreement_candidates': len(picked)})
        if with_text:
            prepare_text(built, data)
        output.extend(built)
    assert len(output) == len(history)
    return output


def prepare_text(rows, data):
    import benchmark_pairwise_replacement_v2 as judge
    needed = {action[k] for row in rows for action in row['treatment']
              if 'mp' not in action for k in ('incumbent', 'challenger')}
    corpus = {}
    if needed:
        with (data / 'documents.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if item['id'] in needed:
                    corpus[item['id']] = item['content']
                    if len(corpus) == len(needed):
                        break
    assert set(corpus) == needed
    for row in rows:
        row['query'] = judge.strip_html(row['query'])
        terms = judge.terms(row['query'])
        row['pairs'] = []
        for index, action in enumerate(row['treatment']):
            if 'mp' in action:
                continue
            left, right = action['incumbent'], action['challenger']
            row['pairs'].append({'index': index,
                                 'incumbent_text': judge.passages(corpus[left], terms)[0],
                                 'challenger_text': judge.passages(corpus[right], terms)[0]})


def run_judge(rows, checkpoint):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import benchmark_pairwise_replacement_v2 as judge
    pending = [r for r in rows if r['pairs']]
    print('Fresh pairs:', sum(len(r['pairs']) for r in pending), flush=True)
    if not pending:
        return {}
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                                dtype=torch.bfloat16).to('cuda').eval()
    token_ids = [tok.encode(s, add_special_tokens=False) for s in 'ABC']
    assert all(len(i) == 1 for i in token_ids)
    stats = judge.score_arm(model, tok, pending, 'direct', 12, 'cuda', [i[0] for i in token_ids])
    for row in pending:
        for item in row['pairs']:
            forward, reverse = [item['pred'][s] for s in ('fwd', 'rev')]
            action = row['treatment'][item['index']]
            action['p_fwd'], action['p_rev'] = forward['challenger'], reverse['challenger']
            action['mp'] = (action['p_fwd'] + action['p_rev']) / 2
            action['agree'] = all(max(v, key=v.get) == 'challenger' for v in (forward, reverse))
    return stats


def evaluate(rows, scored):
    report = {'queries': len(rows), 'coverage': {}, 'scores': {}}
    baseline = bootstrap([(r['domain'], r['baseline']) for r in rows])['point']
    zeroes = [r for r in rows if r['baseline'] <= 1e-12]
    report['baseline_zeroes'] = len(zeroes)
    for name in ('control', 'treatment'):
        maximum = lambda r: max(0., *(a['delta'] for a in r[name]))
        report['coverage'][name] = {
            'oracle': baseline + bootstrap([(r['domain'], maximum(r)) for r in rows])['point'],
            'queries_with_beneficial_action': sum(maximum(r) > 1e-12 for r in rows),
            'zeroes_with_beneficial_action': sum(maximum(r) > 1e-12 for r in zeroes)}
        if scored or name == 'control':
            counts = Counter()
            for row in rows:
                chosen = max(eligible(row[name]), key=lambda a: a['mp'], default=None)
                delta = chosen['delta'] if chosen else 0.
                counts['kept' if not chosen else 'helpful' if delta > 1e-12
                       else 'harmful' if delta < -1e-12 else 'neutral'] += 1
                row[f'{name}_delta'] = delta
            report['scores'][name] = {'ndcg': baseline + bootstrap(
                [(r['domain'], r[f'{name}_delta']) for r in rows])['point'], 'counts': counts}
    if scored:
        report['paired_vs_control'] = bootstrap(
            [(r['domain'], r['treatment_delta'] - r['control_delta']) for r in rows])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--folds', nargs='+', choices=('f0', 'f1', 'f2', 'dev'), required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or ('dev' in args.folds and len(args.folds) != 1):
        parser.error('Use a new output file and report dev separately.')
    rows = [row for fold in args.folds for row in build(args.root, fold, args.model is not None)]
    assert len({(r['domain'], r['qid']) for r in rows}) == len(rows)
    stats = run_judge(rows, args.model) if args.model else {}
    summary = evaluate(rows, args.model is not None)
    output = {'protocol': 'Six pairs. Keep first three original actions and offer up to three '
              'distinct CE top20 challengers outside dense top20, replacing the weakest CE-ranked '
              'incumbent among positions 1–3 only if the CE prefers the challenger. Refill from '
              'the original shortlist. Retain the same model, excerpts and acceptance rule. '
              'Labels score outcomes only.',
              'folds': args.folds, 'summary': summary, 'new_judge_inference': stats,
              'per_query': [{k: v for k, v in row.items() if k not in ('query', 'pairs')} for row in rows]}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
