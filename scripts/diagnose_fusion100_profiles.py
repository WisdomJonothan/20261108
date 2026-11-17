"""Post-hoc dev mechanism analysis only; never selects deployment weights.

python scripts/diagnose_fusion100_profiles.py --data-root D:/RETECO \
  --experiment-root D:/RETECO/results/server_review_20261116
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

ARMS = ['title_ce', 'title_fusion', 'full_ce', 'full_fusion']

def read_run(path):
    result = {}
    for line in path.read_text().splitlines():
        q, _, doc, rank, score, _ = line.split()
        result.setdefault(q, []).append((int(rank), doc, float(score)))
    return {q: sorted(rows) for q, rows in result.items()}

def ndcg(ids, gold):
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return sum((d in gold) / math.log2(i + 2) for i, d in enumerate(ids[:10])) / ideal if ideal else 0.

def z(values):
    v = np.asarray(values, dtype=float)
    return (v - v.mean()) / max(float(v.std()), 1e-8)

def bootstrap(values, draws=10000):
    rng = np.random.default_rng(20261116)
    means = np.array([np.mean(v) for v in values.values()])
    domain = means[rng.integers(0, len(means), (draws, len(means)))].mean(1)
    query = np.zeros(draws)
    for vals in values.values():
        v = np.asarray(vals)
        query += v[rng.integers(0, len(v), (draws, len(v)))].mean(1) / len(values)
    return {'macro': float(means.mean()), 'domain_ci95': np.quantile(domain, [.025,.975]).tolist(),
            'within_domain_query_ci95': np.quantile(query, [.025,.975]).tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--experiment-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = ap.parse_args(); root = args.data_root; exp = args.experiment_root
    domains = sorted(p.name for p in (root/'data/track1_tempo').iterdir() if (p/'qrels_dev.txt').exists())
    trans = {r+'__profile_'+p: {} for r in ARMS for p in ARMS}
    scores = {a+'_'+m: {} for a in ARMS for m in ['ce','fused']}; scores['frozen'] = {}
    details = []; tie_queries = 0; diagonal_max_error = 0.
    for domain in domains:
        gold = {}
        for line in (root/'data/track1_tempo'/domain/'qrels_dev.txt').read_text().splitlines():
            q, _, doc, rel = line.split()
            if int(rel)>0: gold.setdefault(q,set()).add(doc)
        qs = [json.loads(l)['id'] for l in (root/'data/track1_tempo'/domain/'examples_dev.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        dense = read_run(root/'runs'/f'dense_mxbai_{domain}_dev.trec')
        frozen = read_run(root/'runs'/f'sub_sft_e2bc_{domain}_dev.trec')
        runs = {a: {m: read_run(exp/'runs'/f'fusion100_{a}_s17_20261115_{domain}_dev_{m}.trec') for m in ['ce','fused']} for a in ARMS}
        for table in [trans, scores]:
            for key in table: table[key][domain] = []
        for q in qs:
            g = gold.get(q,set()); pool = dense[q][:100]; dense10 = set(x[1] for x in pool[:10])
            dz = dict(zip([x[1] for x in pool], z([x[2] for x in pool])))
            scores['frozen'][domain].append(ndcg([x[1] for x in frozen[q]],g))
            for a in ARMS:
                ce = runs[a]['ce'][q]; ce10 = set(x[1] for x in ce[:10])
                for mode in ['ce','fused']:
                    rows = runs[a][mode][q]; ids = [x[1] for x in rows]; top = set(ids[:10])
                    scores[a+'_'+mode][domain].append(ndcg(ids,g))
                    details.append({'domain':domain,'query':q,'arm':a,'mode':mode,
                      'dense_top10_overlap':len(top & dense10), 'ce_top10_overlap':len(top & ce10),
                      'relevant_enter_vs_dense':len((top-dense10)&g), 'relevant_exit_vs_dense':len((dense10-top)&g),
                      'relevant_enter_vs_ce':len((top-ce10)&g), 'relevant_exit_vs_ce':len((ce10-top)&g)})
            for rank_arm in ARMS:
                ids = [x[1] for x in runs[rank_arm]['ce'][q]]
                for profile_arm in ARMS:
                    profile = sorted([x[2] for x in runs[profile_arm]['ce'][q]], reverse=True)
                    fused = [(dz[d]+s,d) for d,s in zip(ids,z(profile))]
                    if len(set(s for s,d in fused)) < len(fused): tie_queries += 1
                    ordered = [d for s,d in sorted(fused,key=lambda x:(-x[0],x[1]))]
                    value = ndcg(ordered,g)
                    trans[rank_arm+'__profile_'+profile_arm][domain].append(value)
                    if rank_arm == profile_arm:
                        diagonal_max_error = max(diagonal_max_error, abs(value-scores[rank_arm+'_fused'][domain][-1]))
    assert len(domains) == 13 and sum(len(v) for v in scores['frozen'].values()) == 519
    assert diagonal_max_error < 1e-10, 'Native score reconstruction mismatch'
    summary = {}
    fields = ['dense_top10_overlap','ce_top10_overlap','relevant_enter_vs_dense','relevant_exit_vs_dense','relevant_enter_vs_ce','relevant_exit_vs_ce']
    for a in ARMS:
        for mode in ['ce','fused']:
            rows = [v for v in details if v['arm']==a and v['mode']==mode]
            summary[a+'_'+mode] = {f: {'sum':sum(v[f] for v in rows),'query_mean':float(np.mean([v[f] for v in rows])),
                'domain_macro_mean':float(np.mean([np.mean([v[f] for v in rows if v['domain']==d]) for d in domains]))} for f in fields}
    report = {'purpose':'Post-hoc dev mechanism analysis only; no deployment weight selection.',
      'uncertainty_scope':'Exploratory unadjusted percentile confidence intervals; 16-cell transplant matrix is not a preregistered confirmatory test. Ranking retained by original TREC order; profile sorted descending. Boundary counts count query-document relevance incidences, not unique documents.',
      'method':'Retain CE document ordering from rank arm; assign descending CE scores from profile arm by position; population z-normalize scores over the same 100 candidates and add dense z-score with fixed equal weights.',
      'queries':519,'domains':13,'diagonal_max_ndcg_error':diagonal_max_error,'transplant_tied_score_cases':tie_queries,
      'transplants':{k:bootstrap(v) for k,v in trans.items()},
      'profile_swap_delta_vs_native': {r+'__profile_'+p: bootstrap({d:np.array(trans[r+'__profile_'+p][d])-trans[r+'__profile_'+r][d] for d in domains}) for r in ARMS for p in ARMS if r!=p},
      'top10_and_boundary':summary,'per_query_boundary':details}
    final = {'metric':'official domain-macro nDCG@10','queries':519,'domains':13,'seed':17,
      'uncertainty':'Paired bootstraps over domains and within-domain queries; conditional on these trained checkpoints, excludes training randomness. Absolute CIs describe resampling uncertainty, not training reproducibility.',
      'systems':{k:bootstrap(v) for k,v in scores.items()},'comparisons_to_frozen':{},
      'four_arm_contrasts':json.loads((exp/'results/fusion100_four_arm_analysis.json').read_text())['contrasts'],
      'independent_run_audit_and_zero_query_recovery':json.loads((root/'results/fusion100_statistics_review.json').read_text()),
      'frozen_comparison_scope':'Same dense top100 candidate sets. Original training recipe differs; comparison measures deployment performance, not a controlled training-objective effect.'}
    for key in scores:
        if key!='frozen': final['comparisons_to_frozen'][key] = bootstrap({d:np.array(scores[key][d])-scores['frozen'][d] for d in domains})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for filename, obj in [('fusion100_profile_diagnostics.json',report),('fusion100_official_dev_report.json',final)]:
        (args.output_dir/filename).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps({'diagonal_max_error':diagonal_max_error,'transplants':{k:v['macro'] for k,v in report['transplants'].items()},'boundary':summary},indent=2))

if __name__ == '__main__': main()
