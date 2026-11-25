"""Train a small low-rank adapter on held-out-safe RETECO ranking groups."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from math import ceil
from pathlib import Path
import os
import random
from time import perf_counter

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from coverage_cf_bound import read_qrels
from lib_reteco import read_trec_run
from qwen_reranker_adapter import install, save_trainable
from score_specialist_shortlists import TEMPORAL_INSTRUCT, evidence
from run_qwen3_rerank import PREFIX, SUFFIX


def collect_groups(group_root, runs, data, holdout, seed):
    excluded = {(r['domain'], r['qid']) for r in holdout}
    for directory in sorted(d for d in data.iterdir() if d.is_dir() and (d / 'qrels_train.txt').exists()):
        domain = directory.name
        qrels = read_qrels(directory / 'qrels_train.txt')
        dense = read_trec_run(runs / f'dense_mxbai_{domain}_train.trec')
        batch = []
        with (group_root / f'ce_train_groups_{domain}.jsonl').open(encoding='utf-8') as source:
            for line in source:
                group = json.loads(line)
                qid = group['qid']
                if (domain, qid) in excluded:
                    continue
                if group['domain'] != domain or not group['query_full'] or not qrels.get(qid):
                    raise ValueError(f'Invalid training group for {domain}/{qid}')
                order = {doc: i for i, (_, _, doc) in enumerate(dense[qid][:100])}
                picks = {True: [], False: []}
                for item in group['docs']:
                    d = item['id']
                    relevant = d in qrels[qid]
                    if bool(item['label']) != relevant:
                        raise ValueError(f'Training group label differs for {domain}/{qid}')
                    picks[relevant].append(d)
                if not picks[True] or not picks[False]:
                    continue
                ordered = lambda docs: sorted(docs, key=lambda d: (order.get(d, 100), d))
                positive = ordered(picks[True])[:2]
                negative = ordered(picks[False])
                rng = random.Random(f'{seed}:{domain}:{qid}')
                selected = negative[:3]
                if len(negative) > 3:
                    selected.append(rng.choice(negative[3:]))
                docs = positive + selected
                if len(set(docs)) != len(docs):
                    raise ValueError(f'Training group has repeated candidates for {domain}/{qid}')
                batch.append({'qid': qid, 'domain': domain, 'query': group['query_full'],
                              'docs': docs, 'labels': [1]*len(positive) + [0]*len(selected)})
        yield directory, batch


def prepare(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prefix, suffix = (tokenizer.encode(text, add_special_tokens=False) for text in (PREFIX, SUFFIX))
    room = args.max_length - len(prefix) - len(suffix)
    results, clipped, reduced_queries = [], 0, 0
    holdout = load(args.holdout)['queries']
    if len(holdout) != 186 or {r['split'] for r in holdout} != {'train'}:
        raise ValueError('Expected the fixed 186 training-side holdout queries')
    for directory, groups in collect_groups(args.groups_root, args.runs_root, args.data_root, holdout, args.seed):
        needed = {d for r in groups for d in r['docs']}
        corpus = {}
        with (directory / 'documents.jsonl').open(encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                if record['id'] in needed:
                    corpus[record['id']] = record['content']
                    if corpus.keys() == needed:
                        break
        if corpus.keys() != needed:
            raise ValueError(f'Missing documents for {directory.name}')
        for r in groups:
            query = r['query']
            vocabulary = terms(query)
            token_ids = tokenizer.encode(query, add_special_tokens=False)
            max_query = room - 300
            if max_query < 128:
                raise ValueError('The training prompt needs space for the query and an evidence excerpt')
            if len(token_ids) > max_query:
                start = max_query // 2
                query = tokenizer.decode(token_ids[:start]) + ' … ' + tokenizer.decode(token_ids[-(max_query-start-8):])
                reduced_queries += 1
            header = f'<Instruct>: {TEMPORAL_INSTRUCT}\n<Query>: {query}\n<Document>: '
            if len(tokenizer.encode(header, add_special_tokens=False)) >= room:
                raise ValueError(f'Query cannot fit in prompt for {r["qid"]}')
            prompts = [header + evidence(corpus[doc], vocabulary, 'dispersed') for doc in r['docs']]
            tokens = tokenizer(prompts, add_special_tokens=False)['input_ids']
            clipped += sum(len(t) > room for t in tokens)
            results.append({key: r[key] for key in ('domain', 'qid', 'docs', 'labels')} |
                           {'tokens': [prefix + t[:room] + suffix for t in tokens]})
        print(directory.name, len(groups), 'prepared', flush=True)
    keys = {(r['domain'], r['qid']) for r in results}
    if len(keys) != len(results) or keys & {(r['domain'], r['qid']) for r in holdout}:
        raise ValueError('Training groups overlap the holdout or contain duplicate queries')
    model_config = load(args.model / 'config.json')
    report = {'groups': results, 'counts': dict(Counter(r['domain'] for r in results)),
              'clipped_documents': clipped, 'clipped_queries': reduced_queries, 'seed': args.seed,
              'model_type': model_config['model_type'], 'pad': tokenizer.pad_token_id}
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
        file.write('\n')
    print(len(results), 'train groups;', clipped, 'clipped documents;', reduced_queries, 'reduced queries', flush=True)


def collate(rows):
    return rows


class JudgedScore(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        ids = [tokenizer.encode(label, add_special_tokens=False) for label in ('yes', 'no')]
        if any(len(tokens) != 1 for tokens in ids) or ids[0] == ids[1]:
            raise ValueError('Expected distinct yes and no tokens')
        self.register_buffer('decisions', torch.tensor([ids[0][0], ids[1][0]], device=model.device))

    def forward(self, input_ids, attention_mask):
        hidden = self.model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        head = self.model.lm_head.weight.index_select(0, self.decisions)
        value = F.linear(hidden[:, -1, :], head).float()
        return value[:, 0] - value[:, 1]


def loss_for_batch(score_model, tokenizer, batch, weights, device):
    inp = tokenizer.pad({'input_ids': [t for r in batch for t in r['tokens']]},
                        padding=True, return_tensors='pt').to(device)
    values = score_model(**inp)
    sizes = [len(r['labels']) for r in batch]
    losses = []
    for row, value in zip(batch, values.split(sizes), strict=True):
        target = torch.tensor(row['labels'], dtype=torch.float, device=device)
        ranking = -(F.log_softmax(value / 3., dim=0) * target).sum() / target.sum()
        binary = F.binary_cross_entropy_with_logits(value, target)
        losses.append(weights[row['domain']] * (ranking + .25 * binary))
    return torch.stack(losses).mean()


def train(args):
    local = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local)
    device = torch.device(f'cuda:{local}')
    if world > 1:
        dist.init_process_group('nccl')
    leader = local == 0
    torch.manual_seed(args.seed)
    data = load(args.input)
    counts = data['counts']
    records = data['groups']
    if len(records) != sum(counts.values()) or args.model is None:
        raise ValueError('Incorrect prepared data')
    if leader:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier(device_ids=[local])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id != data['pad']:
        raise ValueError('Prepared data and tokenizer do not match')
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                local_files_only=True).to(device).eval()
    if model.config.model_type != data['model_type']:
        raise ValueError('Prepared data has a different model type')
    config = {'layers': args.layers, 'rank': args.rank, 'scale': 2*args.rank}
    install(model, **config)
    judge = JudgedScore(model, tokenizer)
    runner = DistributedDataParallel(judge, device_ids=[local]) if world > 1 else judge
    parameters = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    sampler = DistributedSampler(records, num_replicas=world, rank=local, shuffle=True, seed=args.seed)
    batches = DataLoader(records, batch_size=args.batch_groups, sampler=sampler, collate_fn=collate)
    weights = {d: (len(records)/(len(counts)*n))**.5 for d, n in counts.items()}
    updates_per_epoch = ceil(len(batches) / args.accumulate)
    planned = updates_per_epoch * args.epochs
    started = perf_counter()
    step = 0
    if leader:
        print('Training', len(records), 'groups over', world, 'GPUs;', updates_per_epoch,
              'updates per epoch;', sum(p.numel() for p in parameters), 'trainable parameters', flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        running = torch.zeros((), device=device)
        for micro, r in enumerate(batches):
            slot = micro % args.accumulate
            if slot == 0:
                window = min(args.accumulate, len(batches)-micro)
                optim.zero_grad(set_to_none=True)
            last = slot + 1 == window
            sync = runner.no_sync() if world > 1 and not last else nullcontext()
            with sync:
                loss = loss_for_batch(runner, tokenizer, r, weights, device)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss / window).backward()
            running += loss.detach()
            if not last:
                continue
            step += 1
            warmup = max(1, planned // 20)
            ratio = step / warmup if step <= warmup else max(.10, (planned-step)/(planned-warmup))
            optim.param_groups[0]['lr'] = args.lr * ratio
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optim.step()
            if leader and (step == 1 or step % 20 == 0):
                print('Step', step, '/', planned, 'loss', round(float(loss.detach()), 4),
                      'peak MiB', round(torch.cuda.max_memory_allocated()/2**20),
                      'elapsed seconds', round(perf_counter()-started), flush=True)
        peak = torch.tensor(torch.cuda.max_memory_allocated()/2**20, device=device)
        if world > 1:
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if leader:
            progress = {'epoch': epoch, 'steps': step, 'groups': len(records), 'world_size': world,
                        'parameters': sum(p.numel() for p in parameters), 'loss': float(running)/(world*len(batches)),
                        'seed': args.seed, 'lr': args.lr, 'batch_groups': args.batch_groups,
                        'accumulate': args.accumulate,
                        'max_mib': float(peak), 'seconds': perf_counter()-started}
            path = args.output / f'epoch-{epoch}.pt'
            temporary = path.with_suffix('.tmp')
            save_trainable(model, temporary, config, progress)
            temporary.rename(path)
            print('Checkpoint', path.name, progress, flush=True)
        if world > 1:
            dist.barrier(device_ids=[local])
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare')
    training = commands.add_parser('train')
    for subparser in (prep, training):
        subparser.add_argument('--model', type=Path, required=True)
        subparser.add_argument('--output', type=Path, required=True)
        subparser.add_argument('--seed', type=int, default=2954)
    for flag in ('data-root', 'runs-root', 'groups-root', 'holdout'):
        prep.add_argument(f'--{flag}', type=Path, required=True)
    prep.add_argument('--max-length', type=int, default=1024)
    training.add_argument('--input', type=Path, required=True)
    training.add_argument('--epochs', type=int, default=2)
    training.add_argument('--batch-groups', type=int, default=1)
    training.add_argument('--accumulate', type=int, default=2)
    training.add_argument('--layers', type=int, default=8)
    training.add_argument('--rank', type=int, default=8)
    training.add_argument('--lr', type=float, default=6e-5)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.output.exists():
            parser.error('Use a fresh preparation output')
        prepare(args)
    else:
        if min(args.batch_groups, args.accumulate, args.epochs) < 1:
            parser.error('Training batch, accumulation and epoch counts must be positive')
        train(args)


if __name__ == '__main__':
    main()
