"""Compare fixed calibration policies using shared train-heldout crossfit splits."""
import argparse
import json
from pathlib import Path
import numpy as np
from ir_metrics import ndcg_at_k


def select_alpha(curve):
    return min(range(11), key=lambda i: (-float(curve[i]), round(abs(i / 10 - .5), 12), i)) / 10


def zscore(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(values.std(), 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=4, type=Path, required=True)
    parser.add_argument('--split', type=Path, default=Path('results/fusion100_split.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-policy', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261118)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())['eval']
    expected = {(d, q) for d, queries in split.items() for q in queries}
    assert len(expected) == 214 and len(split) == 13
    domains = sorted(split)
    ids = {d: sorted(split[d]) for d in domains}
    singleton = [d for d in domains if len(ids[d]) == 1]
    evaluated = [d for d in domains if d not in singleton]
    models = {}
    reference_candidates = None
    for path in args.inputs:
        data = json.loads(path.read_text())
        assert data.get('split') == 'train_heldout', 'Input must explicitly declare split=train_heldout'
        tag = data['tag']
        assert tag not in models
        assert 'rows' in data, 'Use original candidate exports for exact shrinkage, not grid approximations'
        if 'rows' in data:
            assert len(data['rows']) == 214
            lookup = {(r['domain'], r['qid']): r for r in data['rows']}
            assert set(lookup) == expected
            candidate_signature = {}
            for key, row in lookup.items():
                candidate_signature[key] = {
                    'gold': row['gold'],
                    'dense': dict(zip(row['docs'], row['dense'])),
                    'docs': row['docs'],
                }
            if reference_candidates is None:
                reference_candidates = candidate_signature
            else:
                for key, signature in candidate_signature.items():
                    reference = reference_candidates[key]
                    assert signature['gold'] == reference['gold'], ('qrels differ', key)
                    assert signature['dense'] == reference['dense'], ('candidate/dense scores differ', key)
                    assert signature['docs'] == reference['docs'], ('candidate order differs; tie policy must match', key)
            table = {}
            for d in domains:
                values = []
                for q in ids[d]:
                    row = lookup[d, q]
                    assert len(row['docs']) == len(set(row['docs'])) == len(row['ce']) == len(row['dense']) == 100
                    ce, dense = zscore(row['ce']), zscore(row['dense'])
                    assert np.isfinite(ce).all() and np.isfinite(dense).all()
                    # Half-grid supports exact fixed shrinkage of every direct grid choice.
                    values.append([ndcg_at_k([row['docs'][i] for i in np.argsort(-(alpha * ce + (1-alpha) * dense), kind='stable')], row['gold'], 10) for alpha in np.arange(21)/20])
                table[d] = np.asarray(values)
            mode = 'exact_alpha_from_candidate_scores'
        models[tag] = {'table': table, 'mode': mode, 'source': str(path)}

    def column(model, alpha):
        return round(alpha * 20)

    def fit(model, masks=None):
        curves = []
        for d in domains:
            t = model['table'][d]
            if masks is not None:
                t = t[masks[d]]
            curve = t.mean(0)
            curves.append(curve[::2] if model['mode'] == 'exact_alpha_from_candidate_scores' else curve)
        direct = select_alpha(np.mean(curves, axis=0))
        return {'equal': .5, 'direct': direct, 'shrink': .5 + .5 * (direct-.5)}

    repeats = {tag: [] for tag in models}
    split_manifest = []
    for repeat in range(20):
        rng = np.random.default_rng(args.seed + repeat)
        folds = {}
        for d in domains:
            order = rng.permutation(len(ids[d]))
            assignment = np.empty(len(order), dtype=int)
            assignment[order] = np.arange(len(order)) % 2
            folds[d] = assignment
        split_manifest.append({d: {q: int(folds[d][i]) for i,q in enumerate(ids[d])} for d in domains})
        for tag, model in models.items():
            totals = {d: {policy: 0. for policy in ['equal','direct','shrink']} for d in evaluated}
            choices = []
            for held in [0,1]:
                masks = {d: np.ones(len(ids[d]),dtype=bool) if d in singleton else folds[d] != held for d in domains}
                policy = fit(model, masks)
                choices.append(policy)
                for d in evaluated:
                    t = model['table'][d][folds[d] == held]
                    for name,alpha in policy.items():
                        totals[d][name] += float(t[:,column(model,alpha)].sum())
            per_domain = {d:{name:value/len(ids[d]) for name,value in v.items()} for d,v in totals.items()}
            macro = {name:float(np.mean([v[name] for v in per_domain.values()])) for name in ['equal','direct','shrink']}
            repeats[tag].append({'repeat':repeat,'macro':macro,'domains':per_domain,'fold_policies':choices,'paired_delta':{'direct_minus_equal':macro['direct']-macro['equal'],'shrink_minus_equal':macro['shrink']-macro['equal'],'shrink_minus_direct':macro['shrink']-macro['direct']}})
    summary = {}
    frozen = {}
    for tag,model in models.items():
        summary[tag] = {name:{'mean':float(np.mean([r['paired_delta'][name] for r in repeats[tag]])),'split_repeat_std':float(np.std([r['paired_delta'][name] for r in repeats[tag]],ddof=1))} for name in repeats[tag][0]['paired_delta']}
        policy = fit(model)
        frozen[tag] = {'requested_alpha':policy,'applied_alpha':{k:column(model,v)/(20 if model['mode']=='exact_alpha_from_candidate_scores' else 10) for k,v in policy.items()},'mode':model['mode'],'source':model['source']}
    metadata = {'split':'train_heldout','seed':args.seed,'repeats':20,'folds':2,'shrink_coefficient':.5,'selection_grid':[i/10 for i in range(11)],'singleton_train_only':singleton,'evaluated_domains':evaluated,'notes':'Paired repeated stratified two-fold crossfit; folds identical across models. Repeats reuse queries and are dependent: split_repeat_std is split sensitivity, not a confidence interval. Checkpoints were selected on these internal queries; this is a conditional calibration diagnostic. Post-dev-unblinding exploratory design: prior seed17/18 dev results were already observed before defining this comparison. This script reads no dev data and fixes shrinkage at .5 without tuning it on this comparison; it is not a pre-dev preregistered confirmatory test.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({**metadata,'summary_by_model_seed':summary,'repeated_results':repeats,'fold_assignments':split_manifest},indent=2))
    args.frozen_policy.parent.mkdir(parents=True,exist_ok=True)
    args.frozen_policy.write_text(json.dumps({**metadata,'fit_queries':214,'policies':frozen},indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
