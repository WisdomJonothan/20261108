# Quarantine: bge-reranker-base runs created by mistake (2026-11-09)

`scripts/phase11_sft_oof.sh` omitted `--model models/bge-reranker-v2-m3`, so the run
tagged `v2m3_pool8_e1b` actually trained **bge-reranker-base** (278M). Its CE score
(0.1957), OOF fusion weights and dev score (0.2799) are invalid and must not be
cited. The run directory was deleted; its rerank / subset / meta artifacts are here
for audit only. The corrected OOF path was abandoned: the 2-epoch arm with an
all-positive weight sweep made an out-of-fold fusion refit unnecessary
(see `results/NEXT_STEPS.md`).
