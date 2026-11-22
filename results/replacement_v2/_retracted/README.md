# Retracted artefacts

| file | reason |
| --- | --- |
| `zero_recall_dev.json` | Produced by `zero_recall.py`, which parsed TREC column 3 (rank) as the score and sorted each list by descending rank, so the "top 100" slice was ranks 901-1000. Its headline ("dense top-100 contains a relevant document for 8 of 138 zero queries") is wrong; the correct value is 86 of 138. Superseded by `results/analysis/SCORE_DIAGNOSIS.json` (`zero_class.zero_depth_in_dense`). |
| `zero_depth_dev.json` (never committed) | Same parsing bug in `zero_depth.py`; withdrawn with the script. |

The corrected analysis and its script live in `results/analysis/`
(`score_diagnosis.py`, `SCORE_DIAGNOSIS.json`); the narrative is
`docs/SCORE_DIAGNOSIS_20261122.md`.
