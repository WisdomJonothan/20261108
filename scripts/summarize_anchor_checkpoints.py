"""Verify completed anchor checkpoints and summarize effective batching."""
import argparse
from collections import Counter
import json
from pathlib import Path

from compare_anchor_reference import load
from replay_anchor_fallback import index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--direct-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = load(args.checkpoint_dir / 'manifest.json')
    final = load(args.direct_report)
    indexed = index(final)
    nominal = manifest['batch_size']
    sizes = Counter()
    retries = []
    prompts, tokens = 0, 0
    if len(manifest['rows']) != final['queries'] or len(indexed) != final['queries']:
        raise ValueError('Incomplete or mismatched evaluation report')
    for i, (domain, qid, fold) in enumerate(manifest['rows']):
        row = load(args.checkpoint_dir / f'{i:04d}.json')
        if (domain, qid) != (row['domain'], row['qid']) or row['scores'] != indexed[domain, qid]['scores']:
            raise ValueError(f'Per-query checkpoint and final scores differ for {domain}/{qid}')
        scored = row['inference']
        batch = scored.get('effective_batch_size', nominal)
        attempts = scored.get('oom_retries', 0)
        sizes[batch] += 1
        prompts += scored['prompts']
        tokens += scored['prompt_tokens']
        if batch != nominal or attempts:
            retries.append({'domain': domain, 'qid': qid, 'fold': fold, 'batch_size': batch,
                            'retries': attempts, 'prompt_tokens': scored['prompt_tokens'],
                            'peak_mb': scored['gpu_peak_mb']})
    if prompts != final['inference']['prompts'] or tokens != final['inference']['prompt_tokens']:
        raise ValueError('Total prompt or token count differs from individual saved queries')
    result = {
        'queries': final['queries'], 'prompts': prompts, 'prompt_tokens': tokens,
        'nominal_batch_size': nominal, 'effective_batch_sizes': dict(sorted(sizes.items())),
        'resumed_queries': final['inference'].get('resumed_queries', 0),
        'retry_queries': retries, 'torch': manifest['torch'], 'transformers': manifest['transformers'],
        'gpu': manifest['gpu'], 'code_sha256': manifest['code'],
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
