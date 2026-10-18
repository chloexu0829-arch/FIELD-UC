"""Matched random-gene-set null, exact Spearman test and Harrell optimism correction."""
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

BINS = ["mean_bin", "variance_bin"]


# ------------------------------------------------------------------ matched null
def expression_deciles(logcpm):
    """Joint mean/variance deciles of discovery log2CPM (genes x samples)."""
    m = pd.DataFrame({"mean": logcpm.mean(axis=1), "variance": logcpm.var(axis=1, ddof=0)})
    m = m.replace([np.inf, -np.inf], np.nan).dropna()
    for col, out in zip(["mean", "variance"], BINS):
        m[out] = pd.qcut(m[col].rank(method="first"), q=10, labels=False).astype(int)
    return m


class MatchedSampler:
    """Draw random genes reproducing the module's joint mean/variance decile counts."""

    def __init__(self, target_bins, candidate_bins):
        self.target = target_bins[BINS].astype(int)
        self.candidates = candidate_bins[BINS].astype(int)
        self.positions = defaultdict(list)
        for i, cell in enumerate(map(tuple, self.target.to_numpy())):
            self.positions[cell].append(i)
        self.pools = {tuple(map(int, cell)): frame.index.to_numpy(dtype=object)
                      for cell, frame in self.candidates.groupby(BINS, sort=False)}

    def sample(self, rng):
        genes, used, deficits = [None] * len(self.target), set(), []
        for cell, positions in self.positions.items():
            pool = self.pools.get(cell, np.array([], dtype=object))
            take = min(len(positions), len(pool))
            chosen = rng.choice(pool, size=take, replace=False).tolist() if take else []
            shuffled = rng.permutation(positions).tolist()
            for pos, gene in zip(shuffled[:take], chosen):
                genes[pos] = str(gene)
                used.add(str(gene))
            deficits += [(pos, cell) for pos in shuffled[take:]]
        remaining = self.candidates[~self.candidates.index.astype(str).isin(used)]
        for pos, (mb, vb) in deficits:   # empty cells: nearest cell in Manhattan distance
            dist = (remaining.mean_bin - mb).abs() + (remaining.variance_bin - vb).abs()
            genes[pos] = str(rng.choice(remaining.index[dist == dist.min()].to_numpy(dtype=object)))
            remaining = remaining.drop(index=genes[pos])
        return genes


def random_module(template, genes, present, rng):
    """Attach permuted real (direction, weight) pairs to random genes, carrying the real dropout pattern."""
    order = rng.permutation(len(template))
    module = pd.DataFrame({"gene": genes, "direction": template.direction.to_numpy()[order],
                           "weight": template.weight.to_numpy()[order]})
    return module[np.asarray(present)[order]]


def empirical_p(observed, null):
    """Upper-tail empirical P with the +1 correction."""
    return (1 + np.count_nonzero(np.asarray(null) >= observed)) / (len(null) + 1)


def exact_spearman(x, y):
    """Spearman rho with a two-sided exact permutation P (all n! orderings)."""
    rho = spearmanr(x, y).statistic
    null = [spearmanr(x, p).statistic for p in permutations(y)]
    return rho, float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


# ------------------------------------------------------------------ prediction models
def _fit(X, y):
    scaler = StandardScaler().fit(X)
    return scaler, LogisticRegression(C=0.5, max_iter=2000).fit(scaler.transform(X), y)


def _predict(model, X):
    return model[1].predict_proba(model[0].transform(X))[:, 1]


def _stratified(rng, y):
    return np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in (0, 1)])


def shrinkage(X, y):
    """Van Houwelingen-Le Cessie heuristic shrinkage (LR chi2 - k) / LR chi2."""
    full = sm.Logit(y, sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")).fit(disp=0, maxiter=200)
    lr = 2 * (full.llf - sm.Logit(y, np.ones((len(y), 1))).fit(disp=0, maxiter=200).llf)
    return (lr - X.shape[1]) / lr


def harrell(X, y, seed, n_boot=1000, n_ci=2000, n_perm=5000):
    """Apparent and optimism-corrected AUC (Harrell bootstrap), calibration slope, Brier score."""
    model = _fit(X, y)
    p = _predict(model, X)
    apparent = roc_auc_score(y, p)
    rng, auc_opt, brier_opt, slopes = np.random.default_rng(seed), [], [], []
    for _ in range(n_boot):
        i = _stratified(rng, y)
        m = _fit(X[i], y[i])
        p_boot, p_orig = _predict(m, X[i]), _predict(m, X)
        auc_opt.append(roc_auc_score(y[i], p_boot) - roc_auc_score(y, p_orig))
        brier_opt.append(brier_score_loss(y[i], p_boot) - brier_score_loss(y, p_orig))
        lp = np.log(np.clip(p_orig, 1e-9, 1 - 1e-9) / np.clip(1 - p_orig, 1e-9, 1 - 1e-9))
        try:
            slopes.append(sm.Logit(y, sm.add_constant(lp)).fit(disp=0, maxiter=100).params[1])
        except Exception:
            pass
    rng = np.random.default_rng(seed + 1)
    lo, hi = np.percentile([roc_auc_score(y[i], p[i]) for i in (_stratified(rng, y) for _ in range(n_ci))], [2.5, 97.5])
    rng = np.random.default_rng(seed + 2)
    p_perm = np.mean([abs(roc_auc_score(rng.permutation(y), p) - 0.5) >= abs(apparent - 0.5) for _ in range(n_perm)])
    optimism = float(np.mean(auc_opt))
    return {"k": X.shape[1], "EPV": y.sum() / X.shape[1], "apparent_auc": apparent, "ci_low": lo, "ci_high": hi,
            "optimism": optimism, "corrected_auc": apparent - optimism, "corrected_ci_low": lo - optimism,
            "corrected_ci_high": hi - optimism, "shrinkage": shrinkage(X, y), "calibration_slope": np.mean(slopes),
            "brier_corrected": brier_score_loss(y, p) - np.mean(brier_opt), "p_permutation": p_perm}
