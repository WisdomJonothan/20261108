"""Compare independent and fusion-aware listwise learning below rank one."""

import argparse
import gzip
import json
from pathlib import Path
import random
from collections import Counter

from replay_conditional_rank_fusion import key, labels


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)['per_query']


def adapt_last_deberta_layers(model):
    import torch
    from torch import nn

    class Projection(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.down = nn.Linear(base.in_features, 8, bias=False, device=base.weight.device)
            self.up = nn.Linear(8, base.out_features, bias=False, device=base.weight.device)
            nn.init.zeros_(self.up.weight)

        def forward(self, values):
            return self.base(values) + self.up(self.down(values))

    model.requires_grad_(False)
    locations = []
    for layer in model.deberta.encoder.layer[-4:]:
        for name in ('query_proj', 'value_proj'):
            attention = layer.attention.self
            projection = Projection(getattr(attention, name))
            setattr(attention, name, projection)
            locations.append((attention, name, projection))

    @torch.no_grad()
    def merge():
        for parent, name, projection in locations:
            delta = projection.up.weight @ projection.down.weight
            projection.base.weight.add_(delta.to(projection.base.weight.dtype))
            setattr(parent, name, projection.base)

    return merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'data-root', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--objective', choices=('independent', 'fusion'), required=True)
    parser.add_argument('--deberta-adapter', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new checkpoint directory')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed = 431
    torch.manual_seed(seed)
    torch.set_num_threads(3)
    rows = read(args.inputs)
    if len(rows) != 450 or {r['panel'] for r in rows} != {'initial-186', 'confirm-155', 'reserve-109'}:
        raise ValueError('The registered training population must contain exactly the 450 fit topics')
    gold = labels(rows, args.data_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16 if args.deberta_adapter else torch.float32
    ).to('cuda').train()
    merge = adapt_last_deberta_layers(model) if args.deberta_adapter else None
    examples = []
    for row in rows:
        current = row['current']
        if row['docs'][:10] != current:
            raise ValueError(f'Unexpected evidence order for {key(row)}')
        target = [doc in gold[key(row)] for doc in current[1:]]
        if not any(target) or all(target):
            continue
        limit = tokenizer.encode(row['query'], add_special_tokens=False, max_length=192, truncation=True)
        question = tokenizer.decode(limit, skip_special_tokens=True)
        encoded = tokenizer([question]*10, row['excerpts'][:10], padding=True, truncation='only_second',
                            max_length=512, return_tensors='pt')
        target = torch.tensor(target, dtype=torch.float32)
        examples.append((row['domain'], encoded, target/target.sum()))
    if not examples:
        raise ValueError('No usable training queries')
    counts = Counter(domain for domain, _, _ in examples)
    domain_weights = {domain: size**-.5 for domain, size in counts.items()}
    normalizer = sum(domain_weights[d] for d, _, _ in examples)/len(examples)
    random.Random(seed).shuffle(examples)
    prior = 1/torch.log2(torch.arange(2, 12, dtype=torch.float32, device='cuda'))
    prior = 3*(prior-prior.mean())/prior.std(correction=0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 1e-4 if args.deberta_adapter else 2e-5
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    updates = 0
    losses = []
    for first in range(0, len(examples), 4):
        group = examples[first:first+4]
        optimizer.zero_grad(set_to_none=True)
        for domain, batch, target in group:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                raw = model(**batch.to('cuda')).logits[:, 0].float()
            standardized = (raw-raw.mean())/raw.std(correction=0).clamp_min(.01)
            score = standardized + prior if args.objective == 'fusion' else standardized
            loss = -(target.to('cuda')*score[1:].log_softmax(dim=0)).sum()
            (loss*domain_weights[domain]/normalizer/len(group)).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        updates += 1
        if updates % 20 == 0:
            print(args.objective, updates, 'loss', round(sum(losses[-80:])/len(losses[-80:]), 5), flush=True)
    if merge:
        merge()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    report = {'objective': args.objective, 'model': args.model.name, 'seed': seed,
              'fit_population': len(rows), 'training_groups': len(examples), 'updates': updates,
              'learning_rate': learning_rate, 'trainable_parameters': sum(p.numel() for p in trainable),
              'domains': counts, 'average_listwise_loss': sum(losses)/len(losses)}
    (args.output/'fit_summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
