#!/usr/bin/env python3
"""Fine-tune a cross-encoder re-ranker on the TRAIN split (listwise softmax).

Training signal
---------------
Each row of `results/ce_train_groups_<domain>.jsonl` is one query group:
all in-candidate positives + hard/random negatives (built by
`build_ce_train_groups.py`). The loss is a listwise softmax over the group's
logits (multi-positive):

    loss = logsumexp(all scores) - logsumexp(positive scores)

The dev split is NEVER read here. Model selection uses held-out *train* queries:
their full top-100 candidate list is re-scored after each epoch and nDCG@10 is
computed with scripts/ir_metrics.py, so the selection signal has the same shape
as the official metric without touching dev.

Preprocessing mirrors `run_chunk_rerank.py` exactly at inference time
(`--q-mode title --q-cap 256 --win 250 --pool max`): here the query is capped
to `--q-cap` tokens and each document contributes its first `--win`-token
window (evaluation still max-pools over all windows).

Usage (pilot, ~minutes):
  python train_ce.py --model models/bge-reranker-base --domains bitcoin economics \
      --tag pilot_rrbase --epochs 2 --val-queries 40

Usage (full):
  python train_ce.py --model models/bge-reranker-v2-m3 --domains bitcoin economics \
      genealogy history hsm monero politics --tag v2m3_e3 --epochs 3
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import read_trec_run  # noqa: E402
from ir_metrics import ndcg_at_k  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auc_score(scores, labels):
    """Within-query AUC of the CE score for gold vs non-gold candidates."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def clean_query(text):
    import re
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = t.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
    return re.sub(r"\s+", " ", t).strip()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_needed_docs(domain, needed):
    """Stream documents.jsonl, keep text only for ids in `needed`."""
    out, path = {}, os.path.join(ROOT, "data", "track1_tempo", domain, "documents.jsonl")
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            if d["id"] in needed:
                out[d["id"]] = d["content"]
    return out


def window_texts(tok, text, win, stride):
    """All overlapping `win`-token windows of a document, decoded back to text.

    Mirrors `run_chunk_rerank.windows_of`, so training (which samples one window
    per document per epoch) and validation (which max-pools over all windows,
    exactly like inference) see the same segmentation."""
    ids = tok(text or "", add_special_tokens=False)["input_ids"]
    if not ids:
        return [""]
    if len(ids) <= win:
        return [tok.decode(ids)]
    out = []
    for start in range(0, len(ids), stride):
        chunk = ids[start:start + win]
        if not chunk:
            break
        out.append(tok.decode(chunk))
        if start + win >= len(ids):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "bge-reranker-base"))
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--groups-prefix", default="results/ce_train_groups")
    ap.add_argument("--run-tag", default="mxbai")
    ap.add_argument("--tag", required=True, help="checkpoint/log sub-directory name")
    ap.add_argument("--outdir", default="results/ce_ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--q-mode", choices=["title", "full"], default="title")
    ap.add_argument("--q-cap", type=int, default=256)
    ap.add_argument("--win", type=int, default=250)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--window-mode", choices=["head", "random", "max"], default="max",
                    help="how a document is represented during training: 'max' max-pools "
                         "over all its windows exactly like inference (pooling parity)")
    ap.add_argument("--max-len", type=int, default=508)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--val-queries", type=int, default=12)
    ap.add_argument("--val-depth", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--limit-groups", type=int, default=0)
    ap.add_argument("--keep-last", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="cap the number of document windows used per training step "
                         "(0 = all windows; a random subset is drawn each epoch)")
    ap.add_argument("--pool-chunk", type=int, default=24,
                    help="windows scored per forward pass when window-mode=max")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze the embedding table and the first N encoder layers")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu":
        dtype = torch.float32
    run_dir = os.path.join(ROOT, args.outdir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.jsonl")

    def log(obj):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=dtype, num_labels=1)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)
    if args.freeze_layers:
        base = getattr(model, "roberta", None) or getattr(model, "bert", None) or model
        emb = getattr(base, "embeddings", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad = False
        layers = getattr(getattr(base, "encoder", None), "layer", [])
        for i, layer in enumerate(layers):
            if i < args.freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log({"event": "freeze", "layers": args.freeze_layers,
             "trainable_params_m": round(n_tr / 1e6, 1)})
    model.train()

    # ---------------------------------------------------------------- data
    groups_by_domain, val_q = {}, {}
    rng = random.Random(args.seed)
    for d in args.domains:
        path = os.path.join(ROOT, f"{args.groups_prefix}_{d}.jsonl")
        gs = read_jsonl(path)
        rng.shuffle(gs)
        n_val = max(1, int(round(len(gs) * args.val_frac)))
        val_q[d] = {g["qid"] for g in gs[:n_val]}
        groups_by_domain[d] = [g for g in gs if g["qid"] not in val_q[d]]
    n_train_groups = sum(len(v) for v in groups_by_domain.values())

    needed = {d: set() for d in args.domains}
    for d in args.domains:
        for g in groups_by_domain[d]:
            needed[d].update(x["id"] for x in g["docs"])
    val_runs, val_gold = {}, {}
    for d in args.domains:
        run = read_trec_run(os.path.join(ROOT, "runs", f"dense_{args.run_tag}_{d}_train.trec"))
        val_runs[d] = {q: [t[2] for t in run[q]][:args.val_depth] for q in val_q[d] if q in run}
        for q in val_runs[d]:
            needed[d].update(val_runs[d][q])
        gold = {}
        with open(os.path.join(ROOT, "data", "track1_tempo", d, "qrels_train.txt"),
                  encoding="utf-8") as f:
            for ln in f:
                qid, _, doc, rel = ln.split()
                if qid in val_q[d]:
                    gold.setdefault(qid, {})[doc] = int(rel)
        val_gold[d] = gold

    t0 = time.time()
    corpus = {d: load_needed_docs(d, needed[d]) for d in args.domains}
    # keep only validation queries that survived the cap
    for d in args.domains:
        qs = list(val_runs[d])
        rng.shuffle(qs)
        val_runs[d] = {q: val_runs[d][q] for q in qs[:args.val_queries]}
    log({"event": "data_loaded", "train_groups": n_train_groups,
         "val_queries": {d: len(val_runs[d]) for d in args.domains},
         "docs_loaded": {d: len(corpus[d]) for d in args.domains},
         "seconds": round(time.time() - t0, 1), "model": args.model,
         "dtype": args.dtype, "grad_ckpt": args.grad_ckpt,
         "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)})

    # window cache: every document's overlapping windows, computed once
    win_cache = {}

    def doc_windows(d, docid):
        key = (d, docid)
        w = win_cache.get(key)
        if w is None:
            w = window_texts(tok, corpus[d].get(docid, ""), args.win, args.stride)
            win_cache[key] = w
        return w

    def doc_win_sample(d, docid, rng_):
        ws = doc_windows(d, docid)
        if args.window_mode == "head" or len(ws) == 1:
            return ws[0]
        return ws[rng_.randrange(len(ws))]

    # ------------------------------------------------------------ optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    groups_per_epoch = n_train_groups
    total_steps = max(1, (groups_per_epoch // args.grad_accum + 1) * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps * args.epochs)
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        w = max(1, int(total_steps * args.warmup_frac))
        if step < w:
            return step / w
        return max(0.0, (total_steps - step) / max(1, total_steps - w))

    sched = LambdaLR(opt, lr_lambda)

    def score_docs(d, qid, qtxt, docids):
        """Score documents exactly like `run_chunk_rerank.py`: every window is
        scored and the document score is the max over its windows."""
        flat, spans = [], []
        for x in docids:
            ws = doc_windows(d, x)
            spans.append((len(flat), len(flat) + len(ws)))
            flat.extend(ws)
        outs = []
        with torch.inference_mode():
            for i in range(0, len(flat), 32):
                chunk = flat[i:i + 32]
                enc = tok([qtxt] * len(chunk), chunk, padding=True,
                          truncation=True, max_length=args.max_len,
                          return_tensors="pt").to(device)
                outs.extend(model(**enc).logits.view(-1).float().cpu().tolist())
        return [max(outs[a:b]) for a, b in spans]

    def validate(epoch):
        was_training = model.training
        model.eval()
        per_dom, all_scores, all_auc = {}, [], []
        for d in args.domains:
            # val queries are absent from the group files, so query text comes
            # from the official examples file
            scores = [(qid, docs) for qid, docs in val_runs[d].items() if docs]
            if not scores:
                continue
            qtext = {}
            with open(os.path.join(ROOT, "data", "track1_tempo", d, "examples_train.jsonl"),
                      encoding="utf-8") as f:
                for ln in f:
                    r = json.loads(ln)
                    if r["id"] in val_runs[d]:
                        full = r["query"]
                        qtext[r["id"]] = clean_query(full.split("<p>")[0]) or clean_query(full) \
                            if args.q_mode == "title" else clean_query(full)
            vals, aucs = [], []
            for qid, docs in scores:
                sc = score_docs(d, qid, qtext.get(qid, ""), docs)
                order = np.argsort(-np.asarray(sc))
                ranked = [docs[i] for i in order]
                gold = val_gold[d].get(qid, {})
                vals.append(float(ndcg_at_k(ranked, gold, 10)))
                lbl = np.array([1 if gold.get(x, 0) > 0 else 0 for x in docs])
                if 0 < lbl.sum() < len(lbl):
                    aucs.append(float(auc_score(np.asarray(sc), lbl)))
            per_dom[d] = float(np.mean(vals)) if vals else float("nan")
            all_scores.extend(vals)
            all_auc.extend(aucs)
        model.train(was_training)
        return {"epoch": epoch, "val_ndcg10": float(np.mean(all_scores)) if all_scores else 0.0,
                "val_auc": float(np.mean(all_auc)) if all_auc else 0.0,
                "per_domain": per_dom, "n_val_queries": len(all_scores)}

    # ------------------------------------------------------------- training
    # epoch-0 validation = the zero-shot reference on exactly the same held-out
    # train queries, so the fine-tuning gain is measured against its own baseline
    v0 = validate(0)
    v0["event"] = "epoch0_zeroshot"
    log(v0)

    best = v0["val_ndcg10"] if v0["val_ndcg10"] == v0["val_ndcg10"] else -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        order = []
        for d in args.domains:
            for g in groups_by_domain[d]:
                order.append((d, g))
        random.Random(args.seed + epoch).shuffle(order)
        if args.limit_groups:
            order = order[: args.limit_groups]
        wrng = random.Random(args.seed * 1000 + epoch)
        ep_t0, ep_loss, ep_n, nb = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (d, g) in enumerate(order):
            docs = g["docs"]
            qtxt = g["query_title"] if args.q_mode == "title" else g["query_full"]
            labels = torch.tensor([x["label"] for x in docs], device=device)
            if args.window_mode == "max":
                flat, spans = [], []
                for x in docs:
                    ws = doc_windows(d, x["id"])
                    if args.max_windows and len(ws) > args.max_windows:
                        idx = sorted(wrng.sample(range(len(ws)), args.max_windows))
                        ws = [ws[i] for i in idx]
                    spans.append((len(flat), len(flat) + len(ws)))
                    flat.extend(ws)
                parts = []
                for j in range(0, len(flat), args.pool_chunk):
                    chunk = flat[j:j + args.pool_chunk]
                    enc = tok([qtxt] * len(chunk), chunk, padding=True,
                              truncation=True, max_length=args.max_len,
                              return_tensors="pt").to(device)
                    parts.append(model(**enc).logits.view(-1).float())
                allsc = torch.cat(parts)
                logits = torch.stack([allsc[a:b].max() for a, b in spans])
            else:
                enc = tok([qtxt] * len(docs),
                          [doc_win_sample(d, x["id"], wrng) for x in docs],
                          padding=True, truncation=True, max_length=args.max_len,
                          return_tensors="pt").to(device)
                logits = model(**enc).logits.view(-1).float()
            pos = labels > 0
            if not bool(pos.any()):
                continue
            loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[pos], dim=0)
            (loss / args.grad_accum).backward()
            ep_loss += float(loss)
            ep_n += 1
            nb += 1
            if nb % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            if args.max_steps and nb >= args.max_steps:
                break
            if nb % 50 == 0:
                log({"event": "step", "epoch": epoch, "step": nb, "loss": round(ep_loss / ep_n, 4),
                     "lr": sched.get_last_lr()[0],
                     "pairs_s": round(sum(len(x["docs"]) for _, x in order[:nb])
                                      / max(1e-9, time.time() - ep_t0), 1),
                     "gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                     if device == "cuda" else None})
        if nb % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        v = validate(epoch)
        v.update({"event": "epoch", "train_loss": round(ep_loss / max(1, ep_n), 4),
                  "train_seconds": round(time.time() - ep_t0, 1),
                  "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                  if device == "cuda" else None})
        log(v)
        # save
        ckpt = os.path.join(run_dir, f"epoch{epoch}")
        model.save_pretrained(ckpt, safe_serialization=True)
        tok.save_pretrained(ckpt)
        if v["val_ndcg10"] > best:
            best = v["val_ndcg10"]
            model.save_pretrained(os.path.join(run_dir, "best"), safe_serialization=True)
            tok.save_pretrained(os.path.join(run_dir, "best"))
            log({"event": "best", "epoch": epoch, "val_ndcg10": best})
        # prune old checkpoints
        for e in range(1, epoch):
            p = os.path.join(run_dir, f"epoch{e}")
            if os.path.isdir(p) and epoch - e > args.keep_last:
                import shutil
                shutil.rmtree(p, ignore_errors=True)
    log({"event": "done", "best_val_ndcg10": best, "run_dir": run_dir})


if __name__ == "__main__":
    main()
