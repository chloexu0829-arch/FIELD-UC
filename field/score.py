"""Frozen single-sample score (equation 3): 0.5 + sum(d * a * (r - 0.5)) / sum(a)."""
import numpy as np
import pandas as pd

import config as C


def score_from_ranks(ranks, module):
    """ranks: within-sample percentile ranks (genes x samples); module: gene, direction, weight."""
    present = module[module.gene.isin(ranks.index)]
    signed = (present.direction.astype(float) * present.weight.astype(float)).to_numpy()
    score = 0.5 + signed @ (ranks.loc[present.gene].to_numpy(dtype=float) - 0.5) / present.weight.abs().sum()
    return pd.Series(np.clip(score, 0.0, 1.0), index=ranks.columns)


def rank_score(expr, module, min_coverage=C.MIN_COVERAGE):
    """expr: genes x samples on any scale. Absent module genes are dropped, not imputed;
    a cohort is scored only if at least `min_coverage` of the module genes are measured."""
    module = module.drop_duplicates("gene")
    coverage = module.gene.isin(expr.index).mean()
    if coverage < min_coverage:
        raise ValueError(f"only {coverage:.0%} of module genes measured")
    return score_from_ranks(expr.rank(axis=0, pct=True), module)
