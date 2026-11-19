# Full-question CE calibration follow-up

This note records seed 17. The [November 19 research update](RESEARCH_UPDATE_20261119.md)
reports the matched title control, seeds 18–19, and calibration stability.

The full-question independently trained CE reaches domain-macro nDCG@10 0.364223 on the official 519-query, 13-domain dev set after calibration. A fixed grid alpha=0,0.1,...,1 was evaluated on the 214-query held-out portion of official train. It selected alpha=0.7 in alpha*z(CE)+(1-alpha)*z(dense). The checkpoint was already fixed by the original internal-validation protocol.

The same checkpoint with alpha=0.5 scores 0.345016. Every query's alpha=0.5 nDCG was reproduced by the independent calibration evaluator. The improvement is +0.019207 with a paired domain-bootstrap 95% interval [0.010905,0.027073], and 11 of 13 domains improve. Relative to the previous frozen system (0.358099), the difference is +0.006125 with interval [-0.005628,0.020003]. These intervals condition on one trained checkpoint and the selected weight. Matched-seed replication and the title-query calibration control are the next experiments.

The z(CE)-only training control scores CE=0.318146 and fused=0.329046. The raw-CE full-question control scores CE=0.330958 and fused=0.345016. Dense-conditioned training scores CE=0.273638 and fused=0.318887. All use the same split and seed 17. The controls motivate studying both normalization and dense conditioning.

To reproduce calibration after scoring the checkpoint, run ops/export_trainval_calibration.py, scripts/calibrate_trainval_alpha.py, then scripts/eval_calibrated_fusion.py. The last script verifies every alpha=0.5 query against the original evaluation before writing calibrated runs. See results/full_ce_calibrated_dev_audited.json and results/full_ce_calibrated_statistics_review.json for complete results. Official data and model checkpoints are obtained separately.
