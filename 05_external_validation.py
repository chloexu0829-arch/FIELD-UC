"""Step 5. Frozen scores in five GEO cohorts, each against 1,000 matched random gene sets.

Scores: original module, 47-gene signature and an unsupervised inflammation reference (Hallmark
Inflammatory Response, all directions +1, weights 1). Nothing is fitted in any validation cohort.
Outputs: results/external_auc.csv, results/matched_null.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

import config as C
from field.geo import COHORTS, load_cohort, patient_level
from field.score import score_from_ranks
from field.stats import MatchedSampler, empirical_p, exact_spearman, expression_deciles, random_module

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scores-only", action="store_true", help="score the frozen modules without the discovery-data matched null")
args = parser.parse_args()
scores = {
    "Original": pd.read_csv(C.RESULTS / "original_module.csv"),
    "FIELD-47": pd.read_csv(C.RESULTS / "field47_signature.csv"),
}
inflammation = pd.DataFrame({"gene": sorted(set(C.HALLMARK_INFLAMMATION.read_text().split())),
                             "direction": 1, "weight": 1.0})
scores["Inflammation reference"] = inflammation

cohorts = {}
for name in COHORTS:
    expr, y, info = load_cohort(name)
    patient = info["patient_id"] if name == "GSE236055" else pd.Series(expr.columns, index=expr.columns)
    cohorts[name] = dict(ranks=expr.rank(axis=0, pct=True), y=y, patient=patient, info=info)
    print(f"{name}: {expr.shape[0]} genes, {int(y.sum())} positive vs {int((1 - y).sum())} reference samples")


def unit_auc(score, c):
    return roc_auc_score(*patient_level(score, c["y"], c["patient"]))


rows = []
for name, c in cohorts.items():
    for score_name, module in scores.items():
        coverage = module.gene.isin(c["ranks"].index).mean()
        rows.append({"cohort": name, "score": score_name, "coverage": coverage,
                     "auc": unit_auc(score_from_ranks(c["ranks"], module), c)})
auc = pd.DataFrame(rows)
assert auc[auc.score != "Inflammation reference"].coverage.min() >= C.MIN_COVERAGE
if args.scores_only:
    auc.to_csv(C.RESULTS / "external_auc_scores_only.csv", index=False)
    print(auc.round(3).to_string(index=False))
    raise SystemExit(0)

from field.data import logcpm, uc_counts

poles = pd.read_csv(C.RESULTS / "poles.csv", index_col=0)
original = pd.read_csv(C.RESULTS / "original_de.csv")
adjusted = pd.read_csv(C.RESULTS / "adjusted_de.csv")
adjusted["in_module"] = adjusted.gene.isin(scores["FIELD-47"].gene)
designs = {  # name: (DE universe, discovery biopsies used to match expression level and variance)
    "Original": (original, poles.index[poles.group.isin(["CRC-like", "UC-like"]) & poles.inflammation.notna()]),
    "FIELD-47": (adjusted, poles.index[poles.group.isin(["CRC-like", "UC-like"]) & poles.cohort.eq("IBDome")
                                       & poles.is_colorectal & poles.local_histology.notna()]),
}

# matched null: same size, permuted real (direction, weight) pairs, joint mean/variance deciles
# in the discovery biopsies, and the same genes missing as in the real module
excluded = set().union(*(set(scores[n].gene) for n in designs))
null_rows, summary = [], []
for m, (name, (de, samples)) in enumerate(designs.items()):
    deciles = expression_deciles(logcpm(uc_counts(samples).T).T)
    template = scores[name].reset_index(drop=True)
    for k, (cohort, c) in enumerate(cohorts.items()):
        present = template.gene.isin(c["ranks"].index).to_numpy()
        pool = sorted((set(de.gene) & set(deciles.index) & set(c["ranks"].index)) - excluded)
        sampler = MatchedSampler(deciles.loc[template.gene], deciles.loc[pool])
        rng = np.random.default_rng(np.random.SeedSequence([C.NULL_SEED, m, k]).generate_state(1, dtype=np.uint32)[0])
        null = [unit_auc(score_from_ranks(c["ranks"], random_module(template, sampler.sample(rng), present, rng)), c)
                for _ in range(C.NULL_N)]
        observed = auc.set_index(["cohort", "score"]).loc[(cohort, name), "auc"]
        null_rows += [{"cohort": cohort, "score": name, "null_auc": v} for v in null]
        summary.append({"cohort": cohort, "score": name, "observed_auc": observed, "null_mean": np.mean(null),
                        "null_q025": np.quantile(null, 0.025), "null_q975": np.quantile(null, 0.975),
                        "p_empirical": empirical_p(observed, null)})
summary = pd.DataFrame(summary)
summary["p_bh"] = summary.groupby("score").p_empirical.transform(lambda p: multipletests(p, method="fdr_bh")[1])
auc = auc.merge(summary.drop(columns="observed_auc"), on=["cohort", "score"], how="left")
auc.to_csv(C.RESULTS / "external_auc.csv", index=False)
pd.DataFrame(null_rows).to_csv(C.RESULTS / "matched_null.csv.gz", index=False)
print(auc.round(3).to_string(index=False))

wide = auc.pivot(index="cohort", columns="score", values="auc").loc[list(COHORTS)]
rho, p = exact_spearman(wide["Original"], wide["Inflammation reference"])
print(f"\nOriginal vs inflammation-reference AUC across cohorts: Spearman rho = {rho:.2f}, exact P = {p:.3f}")

c = cohorts["GSE47908"]                      # original score by reference arm
s = score_from_ranks(c["ranks"], scores["Original"])
arm = c["info"]["disease state"].str.lower()
for ref in ("control", "left-sided", "pancolitis"):
    ref_mask = arm.str.contains(ref)
    print(f"GSE47908 dysplasia vs {ref}: AUC = {roc_auc_score(c['y'][c['y'].eq(1) | ref_mask], s[c['y'].eq(1) | ref_mask]):.3f}")
