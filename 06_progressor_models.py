"""Step 6. GSE236055: 47-gene score versus progressor status at patient level.

Biopsy scores are averaged per patient; progressors already had neoplasia elsewhere in the colon
(cross-sectional design). L2 logistic models with Harrell optimism correction (1,000 bootstraps),
shrinkage, calibration slope and Brier score; then the score within histological strata.
Output: results/gse236055_models.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import config as C
from field.geo import load_cohort
from field.score import rank_score
from field.stats import harrell

SEED = 20260806
expr, y, info = load_cohort("GSE236055")
biopsy = pd.DataFrame({
    "patient": info.patient_id, "y": y, "score": rank_score(expr, pd.read_csv(C.RESULTS / "field47_signature.csv")),
    "age_dx": pd.to_numeric(info.age_at_diagnosis, errors="coerce"),
    "duration": pd.to_numeric(info.disease_duration_yr, errors="coerce"),
    "inflamed": info.inflammation_status.isin(["Chronic", "Chronic_active"]),
})
patients = biopsy.groupby("patient").agg(score=("score", "mean"), y=("y", "max"), age_dx=("age_dx", "mean"),
                                         duration=("duration", "mean"), inflamed_fraction=("inflamed", "mean")).dropna()
yy = patients.y.to_numpy()
print(f"{len(patients)} patients, {yy.sum()} progressors; Mann-Whitney P = "
      f"{mannwhitneyu(patients.score[yy == 1], patients.score[yy == 0]).pvalue:.2g}")

models = {"47-gene score": ["score"], "47-gene score + clinical": ["score", "age_dx", "duration"],
          "clinical + inflammation": ["age_dx", "duration", "inflamed_fraction"]}
table = pd.DataFrame({name: harrell(patients[cols].to_numpy(float), yy, SEED) for name, cols in models.items()}).T
table.to_csv(C.RESULTS / "gse236055_models.csv")
print(table.astype(float).round(3).to_string())

for stratum, mask in [("non-inflamed", ~biopsy.inflamed), ("inflamed", biopsy.inflamed)]:
    agg = biopsy[mask].groupby("patient").agg(score=("score", "mean"), y=("y", "max"))
    pos, neg = agg.score[agg.y == 1].to_numpy(), agg.score[agg.y == 0].to_numpy()
    rng = np.random.default_rng(20261001)
    boots = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                           np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(2000)]
    print(f"{stratum} biopsies ({mask.sum()}; {len(pos)} vs {len(neg)} patients): "
          f"AUC = {roc_auc_score(agg.y, agg.score):.3f} (95% CI {np.percentile(boots, 2.5):.3f}-"
          f"{np.percentile(boots, 97.5):.3f}), P = {mannwhitneyu(pos, neg).pvalue:.2g}")
