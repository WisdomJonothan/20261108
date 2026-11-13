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
import re
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


# ---------------------------------------------------------------- query text
# Four representations of the *same* question under one word budget, so the
# comparison isolates what truncation throws away (see ladder item 3 in
# docs/RESEARCH_PLAN_NEXT.md): the bare title, a fixed-width body window, an
# extractive summary, and a constraint-preserving selection that keeps the
# sentences carrying time ranges / exclusion / comparison cues.
CUE_RE = re.compile(
    r"(before|after|between|during|since|until|prior to|by (the )?(end|time)|"
    r"\b(1[5-9]\d\d|20\d\d)\b|not |exclud|except|without|other than|"
    r"more than|less than|fewer|at least|at most|than|first|last|earliest|latest)",
    re.I)


def _sents(text):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text or "") if s.strip()]


def compose_query(title, body, mode, cap_words):
    """Render one query under `mode`, capped at `cap_words` words (0 = no cap).

    title/body are already tag-stripped. `full` reproduces the historic
    behaviour; `body` is a fixed left window of the body; `ctxt` keeps the
    title then greedily adds body sentences that carry a constraint cue;
    `summ` does the same but ranks sentences by term frequency instead of cues.
    """
    title, body = (title or "").strip(), (body or "").strip()
    if mode == "title":
        txt = title
    elif mode == "full":
        txt = (title + " " + body).strip()
    elif mode == "body":
        txt = body or title
    else:
        keep, used = [title], len(title.split())
        sents = _sents(body)
        if mode == "summ":
            freq = {}
            for w in re.findall(r"[a-z]{4,}", body.lower()):
                freq[w] = freq.get(w, 0) + 1
            order = sorted(range(len(sents)),
                           key=lambda i: -sum(freq.get(w, 0) for w in
                                              set(re.findall(r"[a-z]{4,}", sents[i].lower())))
                           / max(1, len(sents[i].split())))
        else:
            def cue_score(t):
                """How much of a constraint this sentence carries (0 = none).

                Precision matters more than recall here: the point of `ctxt` is
                to add only sentences that hold a condition which the title
                drops, so a bare "first" or a negated verb in ordinary prose
                must not qualify.
                """
                sc = 0.0
                if re.search(r"(\b1[5-9][0-9][0-9]\b|\b20[0-9][0-9]\b)", t):
                    sc += 3
                if re.search(r"\b(before|after|between|during|since|until|prior to|"
                             r"no later than|earlier than|later than|as of)\b", t, re.I):
                    sc += 3
                if re.search(r"(\bexcept\b|\bexcluding\b|\bwithout\b|other than|"
                             r"\bno longer\b|\bnot\b[^.]{0,30}\b(includ|mention|involv|"
                             r"use|used|about|from)\w*)", t, re.I):
                    sc += 2
                if re.search(r"(more than|less than|fewer|at least|at most|"
                             r"\bearliest\b|\blatest\b)", t, re.I):
                    sc += 2
                return sc

            scored = [(cue_score(x), k) for k, x in enumerate(sents)]
            order = [k for sc, k in sorted(scored, key=lambda q: (-q[0], q[1])) if sc >= 2]
        picked = set()
        for i in order:
            n = len(sents[i].split())
            if cap_words and used + n > cap_words:
                continue
            keep.append(sents[i])
            picked.add(i)
            used += n
        txt = " ".join(keep)
    if cap_words:
        txt = " ".join(txt.split()[:cap_words])
    return txt


def split_raw_query(raw):
    """(title, body) of a raw query: the part before the first <p> is the title."""
    parts = (raw or "").split("<p>")
    title = parts[0]
    body = " ".join(parts[1:]) if len(parts) > 1 else ""
    return title, body


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


def dcg_discounts(rank, k):
    """1/log2(rank+2) for rank < k, else 0 (rank is 0-based, sorted desc)."""
    d = torch.zeros(rank.numel(), dtype=torch.float32, device=rank.device)
    keep = rank < k
    d[keep] = 1.0 / torch.log2(rank[keep].float() + 2.0)
    return d


def lambdarank_ndcg_loss(logits, labels, k=10):
    """LambdaRank with nDCG@k swap weights (Burges 2010).

    A pair (i, j) with rel_i > rel_j contributes |Delta nDCG@k| of the swap
    times log(1 + exp(s_j - s_i)). Only pairs whose swap touches the top-k
    contributes, so ordering mistakes inside the top 10 dominate while
    reshuffling below rank k is free. Loss is the weighted mean, keeping the
    gradient scale comparable to the listwise softmax.
    """
    n = logits.numel()
    if n < 2:
        return logits.sum() * 0.0
    rel = labels.float()
    gains = torch.pow(2.0, rel) - 1.0
    order = torch.argsort(logits, descending=True)
    rank = torch.empty(n, dtype=torch.long, device=logits.device)
    rank[order] = torch.arange(n, device=logits.device)
    ideal_rel = torch.sort(rel, descending=True).values[:k]
    ideal_dcg = ((torch.pow(2.0, ideal_rel) - 1.0) *
                 dcg_discounts(torch.arange(ideal_rel.numel(),
                                            device=logits.device), k)).sum().clamp_min(1e-9)
    disc = dcg_discounts(rank, k)
    pair = ((rel.unsqueeze(1) > rel.unsqueeze(0)) &
            (rank.unsqueeze(1) < k) & (rank.unsqueeze(0) < k))
    if not bool(pair.any()):
        return logits.sum() * 0.0
    delta = (gains.unsqueeze(1) - gains.unsqueeze(0)) * \
            (disc.unsqueeze(1) - disc.unsqueeze(0))
    w = (delta.abs() / ideal_dcg) * pair.float()
    diff = logits.unsqueeze(1) - logits.unsqueeze(0)   # s_i - s_j, i ranked above j
    loss = torch.nn.functional.softplus(-diff)
    return (w * loss).sum() / w.sum().clamp_min(1e-9)


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
    ap.add_argument("--q-mode", choices=["title", "full", "body", "ctxt", "summ"],
                    default="title")
    ap.add_argument("--q-cap-words", type=int, default=64,
                    help="word budget shared by the four query variants (0 = uncapped)")
    ap.add_argument("--eval-win-limit", type=int, default=0,
                    help="cap document windows while scoring (0 = all)")
    ap.add_argument("--eval-qvariants", default="",
                    help="comma list of modes: score the val split once per mode and exit")
    ap.add_argument("--init-from", default="",
                    help="start from a saved checkpoint dir instead of --model "
                         "(used for eval-only runs and warm starts)")
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
    ap.add_argument("--dump-val-qids", default=None,
                    help="write the seed-determined train/val split here and exit "
                         "(lets later seeds reuse the exact same held-out queries)")
    ap.add_argument("--dump-val-scores", default=None, action="store_true",
                    help="write per-query validation nDCG@10 per epoch to "
                         "results/val_scores_<tag>_e<epoch>.json so arms can be "
                         "compared with a paired bootstrap")
    ap.add_argument("--val-qids", default=None,
                    help="reuse a split written by --dump-val-qids, so a seed "
                         "replication differs only in initialisation/order")
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
    ap.add_argument("--loss", choices=["possum", "allpos", "ndcg"], default="possum",
                    help="listwise objective: 'possum' = -log sum_(d in P) p(d) "
                         "(the original loss), 'allpos' = mean over positives of "
                         "-log p(d) so every gold document is trained directly, "
                         "'ndcg' = LambdaRank with nDCG@10 swap weights")
    ap.add_argument("--val-metric", choices=["mean", "domain_macro"], default="mean",
                    help="checkpoint selection signal: plain mean over held-out "
                         "train queries, or the domain-macro mean that matches the "
                         "official scoring rule")
    ap.add_argument("--sampler", choices=["uniform", "sqrt"], default="uniform",
                    help="epoch sampling: 'uniform' visits every training group "
                         "once (original behaviour); 'sqrt' draws with probability "
                         "proportional to sqrt(groups in the domain), balancing "
                         "small domains without repeating them to parity")
    ap.add_argument("--step-targets", default="results/ce_step_targets",
                    help="prefix of the per-domain step->gold-doc target files "
                         "written by scripts/build_step_targets.py")
    ap.add_argument("--step-lambda", type=float, default=0.0,
                    help="weight of the evidence-step term added to the objective: "
                         "-mean_j log sum_{d in step j} p(d). 0 disables it; the "
                         "official steps are used for training only, never as "
                         "inference input")
    ap.add_argument("--report-steps", action="store_true",
                    help="compatibility no-op: step diagnostics are always logged "
                         "when --step-targets is given")
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

    init_path = args.init_from or args.model
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.init_from:
        print(json.dumps({"event": "init_from", "path": init_path}), flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        init_path, dtype=dtype, num_labels=1)
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
    pinned = None
    if args.val_qids:
        with open(args.val_qids, encoding="utf-8") as f:
            pinned = json.load(f)
    rng = random.Random(args.seed)
    for d in args.domains:
        path = os.path.join(ROOT, f"{args.groups_prefix}_{d}.jsonl")
        gs = read_jsonl(path)
        rng.shuffle(gs)
        if pinned is not None:
            val_q[d] = set(pinned["exclude"][d])
        else:
            n_val = max(1, int(round(len(gs) * args.val_frac)))
            val_q[d] = {g["qid"] for g in gs[:n_val]}
        groups_by_domain[d] = [g for g in gs if g["qid"] not in val_q[d]]
    n_train_groups = sum(len(v) for v in groups_by_domain.values())

    step_targets = {}
    if args.step_lambda:
        for d in args.domains:
            p = os.path.join(ROOT, f"{args.step_targets}_{d}.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    step_targets[d] = json.load(f)
        log({"event": "step_targets", "domains": len(step_targets),
             "queries": sum(len(v) for v in step_targets.values()),
             "lambda": args.step_lambda})

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
    # keep only validation queries that survived the cap (or the pinned eval list)
    for d in args.domains:
        if pinned is not None:
            keep = [q for q in pinned["eval"].get(d, []) if q in val_runs[d]]
        else:
            qs = list(val_runs[d])
            rng.shuffle(qs)
            keep = qs[:args.val_queries]
        val_runs[d] = {q: val_runs[d][q] for q in keep}

    if args.dump_val_qids:
        out = {"exclude": {d: sorted(val_q[d]) for d in args.domains},
               "eval": {d: list(val_runs[d]) for d in args.domains},
               "seed": args.seed, "val_frac": args.val_frac,
               "val_queries": args.val_queries}
        with open(args.dump_val_qids, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(json.dumps({"event": "dumped_val_qids", "path": args.dump_val_qids,
                          "eval_queries": {d: len(val_runs[d]) for d in args.domains}}))
        return
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
            if args.eval_win_limit and len(ws) > args.eval_win_limit:
                ws = ws[:args.eval_win_limit]
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

    def validate(epoch, qmode=None):
        was_training = model.training
        model.eval()
        per_dom, all_scores, all_auc, per_q = {}, [], [], {}
        # multi-evidence counters live outside the per-domain loop so a run
        # with no scorable validation domain still reports cleanly
        step_hit = step_tot = mp10 = mq = 0
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
                        _t, _b = split_raw_query(r["query"])
                        qtext[r["id"]] = compose_query(clean_query(_t), clean_query(_b),
                                                       qmode or args.q_mode,
                                                       args.q_cap_words)
            vals, aucs = [], []
            for qid, docs in scores:
                sc = score_docs(d, qid, qtext.get(qid, ""), docs)
                order = np.argsort(-np.asarray(sc))
                ranked = [docs[i] for i in order]
                gold = val_gold[d].get(qid, {})
                _nd = float(ndcg_at_k(ranked, gold, 10))
                vals.append(_nd)
                per_q[qid] = _nd
                lbl = np.array([1 if gold.get(x, 0) > 0 else 0 for x in docs])
                if 0 < lbl.sum() < len(lbl):
                    aucs.append(float(auc_score(np.asarray(sc), lbl)))
                # multi-evidence diagnostics: does the model satisfy every
                # annotated step, and does it put several positives in the top
                # ten instead of only the easiest one?
                top10 = set(ranked[:10])
                steps = step_targets.get(d, {}).get(qid) if step_targets else None
                if steps:
                    cand = set(docs)
                    for st in steps:
                        avail = [x for x in st if x in cand]
                        if avail:
                            step_tot += 1
                            step_hit += int(any(x in top10 for x in avail))
                n_pos10 = sum(1 for x in ranked[:10] if gold.get(x, 0) > 0)
                mp10 += n_pos10
                mq += int(n_pos10 >= 2)
            per_dom[d] = float(np.mean(vals)) if vals else float("nan")
            all_scores.extend(vals)
            all_auc.extend(aucs)
        model.train(was_training)
        dom_vals = [v for v in per_dom.values() if v == v]
        diag = {}
        if step_tot:
            diag["step_hit_rate"] = round(step_hit / step_tot, 4)
            diag["step_n"] = step_tot
        if scores:
            diag["pos_in_top10"] = round(mp10 / max(1, len(all_scores)), 4)
            diag["q_with_2pos_top10"] = round(mq / max(1, len(all_scores)), 4)
        if args.dump_val_scores and per_q:
            sp = os.path.join(ROOT, "results",
                              "val_scores_%s_e%d.json" % (args.tag, epoch))
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(per_q, f, indent=1, ensure_ascii=False)
            log({"event": "dumped_val_scores", "path": os.path.basename(sp),
                 "queries": len(per_q)})
        return {**diag, "epoch": epoch, "val_ndcg10": float(np.mean(all_scores)) if all_scores else 0.0,
                "val_ndcg10_domain_macro": float(np.mean(dom_vals)) if dom_vals else 0.0,
                "val_auc": float(np.mean(all_auc)) if all_auc else 0.0,
                "per_domain": per_dom, "n_val_queries": len(all_scores)}

    def sel_metric(v):
        """The number used to pick `best`, following --val-metric."""
        if args.val_metric == "domain_macro":
            return v["val_ndcg10_domain_macro"]
        return v["val_ndcg10"]

    # ------------------------------------------------------------- training
    # epoch-0 validation = the zero-shot reference on exactly the same held-out
    # train queries, so the fine-tuning gain is measured against its own baseline
    if args.eval_qvariants:
        modes = [m.strip() for m in args.eval_qvariants.split(",") if m.strip()]
        out = {}
        for m in modes:
            v = validate(0, qmode=m)
            out[m] = {"domain_macro": round(v["val_ndcg10_domain_macro"], 4),
                      "mean": round(v["val_ndcg10"], 4),
                      "n": v["n_val_queries"],
                      "per_domain": {k: round(x, 4) for k, x in v["per_domain"].items()}}
        log({"event": "qvariants", "cap_words": args.q_cap_words, "results": out})
        return

    v0 = validate(0)
    v0["event"] = "epoch0_zeroshot"
    log(v0)

    best = sel_metric(v0) if sel_metric(v0) == sel_metric(v0) else -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if args.sampler == "sqrt":
            # draw n_train_groups groups with P(domain) ~ sqrt(n_domain): small
            # domains are lifted without being repeated up to parity
            doms = [d for d in args.domains if groups_by_domain[d]]
            w = np.sqrt(np.array([len(groups_by_domain[d]) for d in doms], dtype=float))
            w = w / w.sum()
            pick_rng = np.random.default_rng(args.seed * 7919 + epoch)
            grng = random.Random(args.seed + 313 + epoch)
            picks = pick_rng.choice(len(doms), size=n_train_groups, p=w)
            order = []
            for k in picks:
                d = doms[int(k)]
                gs = groups_by_domain[d]
                order.append((d, gs[grng.randrange(len(gs))]))
        else:
            order = []
            for d in args.domains:
                for g in groups_by_domain[d]:
                    order.append((d, g))
            random.Random(args.seed + epoch).shuffle(order)
        if args.limit_groups:
            order = order[: args.limit_groups]
        wrng = random.Random(args.seed * 1000 + epoch)
        ep_t0, ep_loss, ep_n, nb = time.time(), 0.0, 0, 0
        ep_step, ep_step_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (d, g) in enumerate(order):
            docs = g["docs"]
            if args.q_mode == "full":
                qtxt = g["query_full"]
            else:
                _t = g["query_title"] or ""
                _f = g["query_full"]
                _b = _f[len(_t):] if _f.startswith(_t) else _f
                qtxt = compose_query(_t, _b, args.q_mode, args.q_cap_words)
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
            if args.loss == "possum":
                loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[pos], dim=0)
            elif args.loss == "allpos":
                logp = logits - torch.logsumexp(logits, dim=0)
                loss = -logp[pos].mean()
            else:
                loss = lambdarank_ndcg_loss(logits, labels, k=10)
            if args.step_lambda and d in step_targets:
                st = step_targets[d].get(g["qid"])
                if st:
                    idx = {x["id"]: i for i, x in enumerate(docs)}
                    lp = logits - torch.logsumexp(logits, dim=0)
                    terms = []
                    for s in st:
                        hit = [idx[k] for k in s if k in idx]
                        if hit:
                            terms.append(torch.logsumexp(lp[hit], dim=0))
                    if terms:
                        step_loss = -torch.stack(terms).mean()
                        loss = loss + args.step_lambda * step_loss
                        ep_step += float(step_loss)
                        ep_step_n += 1
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
        if args.step_lambda:
            v["step_loss"] = round(ep_step / max(1, ep_step_n), 4)
        log(v)
        # save
        ckpt = os.path.join(run_dir, f"epoch{epoch}")
        model.save_pretrained(ckpt, safe_serialization=True)
        tok.save_pretrained(ckpt)
        if sel_metric(v) > best:
            best = sel_metric(v)
            model.save_pretrained(os.path.join(run_dir, "best"), safe_serialization=True)
            tok.save_pretrained(os.path.join(run_dir, "best"))
            log({"event": "best", "epoch": epoch, "metric": args.val_metric,
                 "score": round(best, 6), "val_ndcg10": round(v["val_ndcg10"], 6),
                 "val_ndcg10_domain_macro": round(v["val_ndcg10_domain_macro"], 6)})
        # prune old checkpoints
        for e in range(1, epoch):
            p = os.path.join(run_dir, f"epoch{e}")
            if os.path.isdir(p) and epoch - e > args.keep_last:
                import shutil
                shutil.rmtree(p, ignore_errors=True)
    log({"event": "done", "best_val_ndcg10": best, "run_dir": run_dir})


if __name__ == "__main__":
    main()
