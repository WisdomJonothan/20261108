"""Audit when truncated reciprocal-rank fusion excludes single-vote candidates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile

from eval_qwen_reranker_adapter import POLICIES, blend
from official.scorer import load_qrels
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from replay_wide_pointwise_vote import retained_candidate_list


def load_parts(path):
    parts = {}
    with tarfile.open(path, 'r|gz') as stream:
        for item in stream:
            name = item.name.removeprefix('./')
            if name in ('consensus.json', 'scopes.json', 'result.json') or (
                name.endswith('.json') and name.startswith(('old-pointwise-p', 'pointwise-p', 'proof-p'))
            ):
                parts[name] = json.load(stream.extractfile(item))
    return parts


def by_topic(rows):
    return {(r['domain'], r['qid']): r for r in rows}


def read_votes(parts):
    pointwise, proof = {}, {}
    for name, payload in sorted(parts.items()):
        if name.startswith('proof-p'):
            shard = by_topic(payload['per_query'])
            if proof.keys() & shard.keys():
                raise ValueError('Proof shards must be query-disjoint')
            proof.update({k: r['observations'] for k, r in shard.items()})
        elif name.startswith(('old-pointwise-p', 'pointwise-p')):
            for key, row in by_topic(payload['per_query']).items():
                cache = pointwise.setdefault(key, {})
                if not cache.keys().isdisjoint(row['scores']):
                    raise ValueError(f'The pointwise score shards overlap for {key}')
                cache.update(row['scores'])
    return pointwise, proof


def summarize(parts, data_root):
    consensus = by_topic(parts['consensus.json']['per_query'])
    scopes = by_topic(parts['scopes.json']['queries'])
    result = by_topic(parts['result.json']['per_query'])
    pointwise, proof = read_votes(parts)
    if not (consensus.keys() == scopes.keys() == result.keys() == pointwise.keys() == proof.keys()):
        raise ValueError('Scores, rankings and candidate lists must align')
    summary = Counter()
    others_hist, all_hist = Counter(), Counter()
    labels = {}
    for key, row in consensus.items():
        original, expanded = (scopes[key][field] for field in ('original', 'wide'))
        old = set(original)
        votes = row['rankings']
        other = [votes[name] for name in ('current', 'reasoned', 'verified')]
        if any(len(vote) != 10 or len(set(vote)) != 10 or set(vote)-old for vote in other):
            raise ValueError(f'Expected three original-pool top-ten votes for {key}')
        revised = retained_candidate_list(original, expanded)
        changed = blend(revised, {}, pointwise[key], POLICIES['adaptive30_free'])
        actual = result[key]['rankings']['local_candidate_scale']
        if head(reciprocal(other+[changed], 60), votes['pointwise'], proof[key], True, .5) != actual:
            raise ValueError(f'The candidate vote must reproduce the reported ranking for {key}')
        counter = Counter(doc for vote in other for doc in vote)
        repeated_other = sum(v >= 2 for v in counter.values())
        counter.update(changed)
        repeated_all = sum(v >= 2 for v in counter.values())
        others_hist[repeated_other] += 1
        all_hist[repeated_all] += 1
        outs = set(changed)-old
        if repeated_all >= 10 and set(actual)-old:
            raise ValueError(f'A single-vote outsider cannot outrank ten repeated documents for {key}')
        summary.update(
            queries=1, repeated_ten_under_other_votes=repeated_other >= 10,
            repeated_ten_under_all_votes=repeated_all >= 10,
            outside_original60_got_new_vote=bool(outs),
            outside_original60_received_new_votes=len(outs),
            outside_original60_got_full_top10=bool(set(actual)-old),
            outside_original60_voted_and_repeat_barrier=bool(outs) and repeated_all >= 10,
            reasoned_equals_verified_set=set(other[1]) == set(other[2]),
        )
        if data_root:
            domain, qid = key
            split = result[key]['split']
            if (domain, split) not in labels:
                labels[domain, split] = load_qrels(data_root / domain / f'qrels_{split}.txt')
            gold = labels[domain, split][qid]
            summary.update(
                outside_positive_in_wide60=bool((set(expanded)-old) & gold),
                outside_positive_in_revised30=bool((set(revised)-old) & gold),
                outside_positive_in_pointwise_top10=bool(outs & gold),
                outside_positive_in_final_top10=bool((set(actual)-old) & gold),
            )
    return {'counts': dict(summary), 'two_vote_histogram_other_three': dict(others_hist),
            'two_vote_histogram_revised_four': dict(all_hist), 'pointwise_vote_depth': 10, 'offset': 60}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(load_parts(args.archive), args.data_root)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
