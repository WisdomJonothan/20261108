"""Continue the same anchor scoring protocol with per-query CUDA OOM recovery."""
import sys

import torch

import eval_anchor_reranker as runner


def install_oom_retry():
    original = runner.judge.score_arm

    def score(model, tokenizer, rows, policy, batch_size, device, option_ids, **kwargs):
        if len(rows) != 1:
            raise ValueError('OOM recovery requires the per-query checkpoint protocol')
        row = rows[0]
        current = batch_size
        retries = 0
        while True:
            try:
                stats = original(model, tokenizer, rows, policy, current, device, option_ids, **kwargs)
            except torch.OutOfMemoryError:
                if current == 1:
                    raise
            else:
                stats['effective_batch_size'] = current
                stats['oom_retries'] = retries
                return stats
            torch.cuda.empty_cache()
            for pair in row['pairs']:
                pair.pop('pred', None)
            reduced = max(1, current // 2)
            print(f'OOM retry for {row["domain"]}/{row["qid"]}: batch {current} -> {reduced}', flush=True)
            current = reduced
            retries += 1

    runner.judge.score_arm = score


if __name__ == '__main__':
    if not any(a == '--checkpoint-dir' or a.startswith('--checkpoint-dir=') for a in sys.argv[1:]):
        raise SystemExit('Pass --checkpoint-dir when running the OOM recovery entry point.')
    install_oom_retry()
    runner.main()
