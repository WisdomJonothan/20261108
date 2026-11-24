"""Reconstruct the actual judge inputs for relevant suffix challengers."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from compare_anchor_reference import load
from coverage_cf_bound import read_steps
from eval_joint_swap_support import prepare
from replay_position_discount import eligible


def read_documents(path, wanted):
    found = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['id'] in wanted:
                found[record['id']] = record['content']
                if len(found) == len(wanted):
                    break
    if found.keys() != wanted:
        raise ValueError(f'Missing {len(wanted - found.keys())} documents in {path}')
    return found


def collect(root, data_root, direct):
    from benchmark_pairwise_replacement_v2 import passages, strip_html, terms

    split = 'dev' if direct['folds'] == ['dev'] else 'train'
    by_domain = defaultdict(list)
    for row in prepare(root, data_root, direct, direct['folds']):
        for action, labels in zip(row['arms']['supervised'][3:], row['treatment']):
            if labels['gold_outcome'] == 'challenger':
                by_domain[row['domain']].append((row, action))
    cases = []
    for domain, pairs in sorted(by_domain.items()):
        folder = data_root / domain
        needed = {a[key] for _, a in pairs for key in ('incumbent', 'challenger')}
        text = read_documents(folder / 'documents.jsonl', needed)
        steps = read_steps(folder / f'qrels_steps_{split}.txt')
        step_rows = {}
        sought = {r['qid'] for r, _ in pairs}
        with (folder / f'steps_{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if record['id'] in sought:
                    step_rows[record['id']] = {
                        step['step_id'].rsplit('_step', 1)[1]: step['step_instruction']
                        for step in record['steps']
                    }
        for row, action in sorted(pairs, key=lambda item: (item[0]['qid'], item[1]['challenger'])):
            query = strip_html(row['query'])
            qid = row['qid']
            excerpts = {key: passages(text[action[key]], terms(query))[0] for key in ('incumbent', 'challenger')}
            full_text = ' '.join(strip_html(text[action['challenger']]).split())
            labeled_steps = [
                {'step': step, 'instruction': step_rows.get(qid, {}).get(step)}
                for step, docs in steps.get(qid, {}).items() if action['challenger'] in docs
            ]
            case_id = f"{domain}/{qid}/{action['challenger'].split('/')[-1]}"
            cases.append({
                'case_id': case_id, 'domain': domain, 'qid': qid,
                'accepted': bool(eligible([action])),
                'action': {k: action[k] for k in ('slot', 'incumbent', 'challenger', 'delta',
                                                 'p_fwd', 'p_rev', 'mp', 'agree')},
                'query': query, 'challenger_text': full_text,
                'challenger_excerpt': excerpts['challenger'], 'incumbent_excerpt': excerpts['incumbent'],
                'challenger_words': len(full_text.split()), 'excerpt_words': len(excerpts['challenger'].split()),
                'full_text_in_excerpt': full_text == excerpts['challenger'],
                'labeled_steps': labeled_steps,
            })
    return cases


def metadata(cases):
    items = []
    for case in cases:
        record = {key: value for key, value in case.items()
                  if key not in ('query', 'challenger_text', 'challenger_excerpt', 'incumbent_excerpt', 'labeled_steps')}
        record['labeled_steps'] = [s['step'] for s in case['labeled_steps']]
        record['excerpt_sha256'] = hashlib.sha256(case['challenger_excerpt'].encode()).hexdigest()
        items.append(record)
    return {'pairs': len(items), 'accepted': sum(r['accepted'] for r in items),
            'rejected': sum(not r['accepted'] for r in items), 'cases': items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--direct-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-review', type=Path, help='Optional private review file with corpus text.')
    args = parser.parse_args()
    data_root = args.data_root or args.root / 'data/track1_tempo'
    cases = collect(args.root, data_root, load(args.direct_report))
    report = metadata(cases)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    if args.local_review:
        with args.local_review.open('x', encoding='utf-8') as stream:
            json.dump(cases, stream, ensure_ascii=False, indent=1)
    print({**{key: report[key] for key in ('pairs', 'accepted', 'rejected')},
           'rejected_full_in_excerpt': sum(not r['accepted'] and r['full_text_in_excerpt'] for r in cases)}, flush=True)


if __name__ == '__main__':
    main()
