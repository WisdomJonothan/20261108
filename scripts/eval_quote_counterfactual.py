"""Test whether a copied quote carries the relevance signal for its document."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_evidence_reasoner import INSTRUCTION
from eval_temporal_condition_intervention import evaluate
from eval_temporal_evidence_categories import ranking
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from replay_stronger_evidence_judge import observations
from score_specialist_shortlists import evidence


ARMS = ('current', 'quote_sufficiency', 'quote_with_paired_ablation')
VIEWS = ('quoted', 'without_quote', 'without_control')


def views(passage, generated, key):
    found = re.search(r'(?:^|\n)\s*Quote:\s*([^\r\n]+)', generated, flags=re.IGNORECASE)
    wanted = re.findall(r'\b\w+\b', found[1].casefold()) if found else []
    if len(wanted) < 5:
        return None
    matches = list(re.finditer(r'\b\w+\b', passage))
    words = [m[0].casefold() for m in matches]
    length = len(wanted)
    hits = [i for i in range(len(words)-length+1) if words[i:i+length] == wanted]
    if len(hits) != 1:
        return None
    start, = hits
    end = start + length
    # Require a nonoverlapping span of the same length in one of the original excerpts.
    candidates = [i for i in range(len(words)-length+1) if (i+length <= start or i >= end)
                  and '[Additional excerpt]' not in passage[matches[i].start():matches[i+length-1].end()]]
    quote_start, quote_end = matches[start].start(), matches[end-1].end()
    quoted = passage[quote_start:quote_end]
    if not candidates or '[Additional excerpt]' in quoted:
        return None
    offset = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % len(candidates)
    control = candidates[offset]
    control_start, control_end = matches[control].start(), matches[control+length-1].end()
    return (quoted, passage[:quote_start] + '[Removed]' + passage[quote_end:],
            passage[:control_start] + '[Removed]' + passage[control_end:])


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identity, cache = observations(args.evidence)
    if identity != (args.model.name, INSTRUCTION, 'dispersed'):
        raise ValueError('The ablations require the same original evidence model and rubric')
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=True)
    tokens = [tokenizer.encode(d, add_special_tokens=False) for d in ('0', '1', '2', '3')]
    if any(len(t) != 1 for t in tokens) or len({t[0] for t in tokens}) != 4:
        raise ValueError('Expected four distinct ordinal-rating tokens')
    ids = [t[0] for t in tokens]
    tiers = torch.arange(4, device='cuda', dtype=torch.float32)
    groups = defaultdict(list)
    source = load(args.manifest)['queries'][args.shard_index::args.shards]
    for row in source:
        groups[row['domain']].append(row)
    output = []
    with torch.inference_mode():
        for domain, rows in sorted(groups.items()):
            texts = read_documents(args.data_root/domain/'documents.jsonl',
                                   {d for row in rows for d in row['docs']})
            pending = []
            for row in rows:
                qid, question = row['qid'], row['query']
                old = cache[domain, qid]
                if old.keys() != set(row['docs']):
                    raise ValueError(f'Unexpected document identities for {domain}/{qid}')
                output.append({'domain': domain, 'qid': qid, 'views': {}})
                vocabulary = terms(question)
                for doc in row['docs']:
                    if not old[doc]['verified']:
                        continue
                    passage = evidence(texts[doc], vocabulary, 'dispersed')
                    spans = views(passage, old[doc]['response'], f'{domain}/{qid}/{doc}')
                    if spans is None:
                        continue
                    values = {}
                    output[-1]['views'][doc] = values
                    for view, text in zip(VIEWS, spans, strict=True):
                        message = f'{INSTRUCTION}\n\nQuestion\n{question}\n\nDocument\n{text}'
                        prompt = tokenizer.apply_chat_template([{'role':'user','content':message}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        pending.append((values, view, prompt + 'Rating: '))
            for start in range(0, len(pending), args.batch_size):
                block = pending[start:start+args.batch_size]
                batch = tokenizer([p for _, _, p in block], padding=True,
                                  add_special_tokens=False, return_tensors='pt').to('cuda')
                raw = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float()
                for (values, view, _), grade in zip(block, (raw.softmax(-1) @ tiers).tolist(), strict=True):
                    values[view] = grade
            print(domain, len(output), len(source), len(pending), flush=True)
    return {'model': identity[0], 'instruction': INSTRUCTION, 'per_query': output}


def replay(args):
    manifest = index({'per_query': load(args.manifest)['queries']})
    _, cached = observations(args.evidence)
    trial = index({'per_query': [r for p in args.views for r in load(p)['per_query']]})
    consensus, frozen = (index(load(path)) for path in (args.consensus, args.reference))
    if not manifest.keys() == cached.keys() == trial.keys() == consensus.keys() == frozen.keys():
        raise ValueError('Ablation comparisons require identical query IDs')
    rows = []
    for key, row in manifest.items():
        docs, old, measured = row['docs'], cached[key], trial[key]['views']
        if set(docs) != old.keys() or not measured.keys() <= {d for d in docs if old[d]['verified']}:
            raise ValueError(f'Mismatched quote ablations for {key}')
        original = consensus[key]['rankings']
        first = frozen[key]['rankings']['verified_0.5']
        if head(reciprocal([original[n] for n in ('current','reasoned','verified','pointwise')], 60),
                original['pointwise'], old, True, .5) != first:
            raise ValueError(f'Unexpected reference ranking for {key}')
        scores = [old[d]['reasoned' if old[d]['verified'] else 'direct'] for d in docs]
        if ranking(docs, scores) != original['verified']:
            raise ValueError(f'The unchanged evidence vote differs for {key}')
        orders = {'current': first}
        for method in ARMS[1:]:
            values = scores.copy()
            for i, d in enumerate(docs):
                if d in measured:
                    entry = measured[d]
                    if set(entry) != set(VIEWS):
                        raise ValueError(f'Incomplete paired views for {key} / {d}')
                    sufficiency = min(values[i], entry['quoted'])
                    if method == 'quote_with_paired_ablation' and entry['without_quote'] >= entry['without_control']:
                        sufficiency = min(sufficiency, old[d]['direct'])
                    values[i] = sufficiency
            vote = ranking(docs, values)
            votes = [original['current'], original['reasoned'], vote, original['pointwise']]
            orders[method] = head(reciprocal(votes, 60), original['pointwise'], old, True, .5)
        rows.append({k: row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                    {'rankings': orders, 'paired_documents': len(measured)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    scoring, replaying = (stages.add_parser(name) for name in ('score', 'evaluate'))
    for stage in (scoring, replaying):
        for name in ('manifest', 'data-root', 'output'):
            stage.add_argument(f'--{name}', type=Path, required=True)
        stage.add_argument('--evidence', type=Path, nargs='+', required=True)
    scoring.add_argument('--model', type=Path, required=True)
    for name, value in (('batch-size', 8), ('shards', 1), ('shard-index', 0)):
        scoring.add_argument(f'--{name}', type=int, default=value)
    for name in ('reference', 'consensus', 'selection', 'reserve-ids'):
        replaying.add_argument(f'--{name}', type=Path, required=name in ('reference', 'consensus'))
    replaying.add_argument('--views', type=Path, nargs='+', required=True)
    replaying.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or (args.stage == 'evaluate' and args.confirmation and not args.selection):
        parser.error('Use a fresh output and supply a frozen choice for confirmation')
    if args.stage == 'score' and not (0 <= args.shard_index < args.shards and args.batch_size > 0):
        parser.error('Check the shard index and batch size')
    result = score(args) if args.stage == 'score' else evaluate(args, replay, ARMS)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(result, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
