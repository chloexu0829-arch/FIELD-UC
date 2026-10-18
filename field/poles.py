"""Clustering of the latent space and within-UC pole assignment (CRC-like / UC-like)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

NEOPLASTIC = ("CRC_CA", "CRC_CMA")


def _explained_variance(Z, labels):
    within = sum(((Z[labels == k] - Z[labels == k].mean(0)) ** 2).sum() for k in np.unique(labels))
    return float(1.0 - within / (((Z - Z.mean(0)) ** 2).sum() + 1e-8))


def _pac(Z, K, n_resamples=50, seed=42):
    """Proportion of ambiguous clustering (consensus in (0.1, 0.9)) over bootstrap K-means."""
    rng, n = np.random.RandomState(seed), len(Z)
    together, sampled = np.zeros((n, n)), np.zeros((n, n))
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        labels = KMeans(K, n_init=5, random_state=rng.randint(1_000_000)).fit(Z[idx]).labels_
        member = np.zeros((n, K))             # draws of each sample per cluster (with multiplicity)
        np.add.at(member, (idx, labels), 1)
        together += member @ member.T
        draws = np.bincount(idx, minlength=n).astype(float)
        sampled += np.outer(draws, draws)
    np.fill_diagonal(sampled, np.maximum(sampled.diagonal(), 1))
    off = (together / np.maximum(sampled, 1))[np.triu_indices(n, k=1)]
    return float(((off > 0.1) & (off < 0.9)).mean())


def select_k(Z, k_range=range(2, 7), seed=42):
    """Borda rank-sum over silhouette, Calinski-Harabasz, seed stability (ARI), explained variance, PAC."""
    rows = []
    for K in k_range:
        runs = [KMeans(K, n_init=20, random_state=s).fit(Z).labels_ for s in range(8)]
        rows.append({
            "K": K,
            "silhouette": np.mean([silhouette_score(Z, r) for r in runs]),
            "calinski_h": np.mean([calinski_harabasz_score(Z, r) for r in runs]),
            "stability_ari": np.mean([adjusted_rand_score(runs[i], runs[j])
                                      for i in range(8) for j in range(i + 1, 8)]),
            "evr": np.mean([_explained_variance(Z, r) for r in runs]),
            "pac": _pac(Z, K, seed=seed),
        })
    table = pd.DataFrame(rows)
    ranks = [table[c].rank(ascending=False) for c in ("silhouette", "calinski_h", "stability_ari", "evr")]
    table["borda_total"] = sum(ranks) + table["pac"].rank(ascending=True)
    best = table.loc[table.borda_total.idxmin()]
    K = int(best.K)
    if best.pac > 0.5:                           # no detectable structure
        return 1, table
    rng = np.random.RandomState(seed)            # bootstrap stability gate
    full = KMeans(K, n_init=20, random_state=seed).fit_predict(Z)
    aris = []
    for _ in range(100):
        idx = rng.choice(len(Z), size=len(Z), replace=True)
        boot = KMeans(K, n_init=5, random_state=rng.randint(1_000_000)).fit_predict(Z[idx])
        aris.append(adjusted_rand_score(full[idx], boot))
    stable = np.mean(aris) > 0.5 and np.mean(aris) - np.std(aris) > 0.3
    return (K if stable else 1), table


def kmedoids(Z, k, n_init=5, max_iter=200, random_state=42):
    """PAM K-medoids (Euclidean) with k-medoids++ initialisation; returns labels and medoid indices."""
    D = pairwise_distances(Z).astype(np.float32)
    n, rng = len(D), np.random.RandomState(random_state)
    best = (np.inf, None, None)
    for _ in range(n_init):
        medoids = [rng.randint(n)]
        for _ in range(k - 1):
            d = D[:, medoids].min(axis=1)
            medoids.append(int(rng.choice(n, p=d / (d.sum() + 1e-10))))
        medoids = sorted(set(medoids))
        while len(medoids) < k:
            c = int(rng.randint(n))
            if c not in medoids:
                medoids.append(c)
        for _ in range(max_iter):
            labels = D[:, medoids].argmin(axis=1)
            cost = D[np.arange(n), np.array(medoids)[labels]].sum()
            improved = False
            for i in range(k):                   # best swap of medoid i with any non-medoid
                others = np.setdiff1d(np.arange(n), medoids)
                candidates = np.tile(medoids, (len(others), 1))
                candidates[:, i] = others
                costs = D[:, candidates].min(axis=2).sum(axis=0)
                j = int(costs.argmin())
                if costs[j] < cost - 1e-5:
                    medoids[i] = int(others[j])
                    labels = D[:, medoids].argmin(axis=1)
                    cost = D[np.arange(n), np.array(medoids)[labels]].sum()
                    improved = True
            if not improved:
                break
        if cost < best[0]:
            best = (cost, labels.copy(), list(medoids))
    return best[1], best[2]


def cluster_poles(latent):
    """Majority rule: a cluster is a CRC pole if most of its members are carcinomas, else a UC pole."""
    frac = latent.disease_status.isin(NEOPLASTIC).groupby(latent.cluster).mean()
    return frac, latent.cluster.map(lambda c: "CRC" if frac[c] > 0.5 else "UC")


def knn_crc_likeness(latent, z_cols, query, reference_crc, k=20):
    """Leave-one-out fraction of carcinomas among the k nearest reference samples (CRC + UC biopsies)."""
    X = StandardScaler().fit_transform(latent[z_cols].values)
    ref = (reference_crc | query).values
    ref_pos = np.flatnonzero(ref)
    is_crc = reference_crc.values[ref].astype(float)
    _, nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X[ref]).kneighbors(X[query.values])
    out = []
    for q, row in zip(np.flatnonzero(query.values), nbrs):
        self_pos = np.searchsorted(ref_pos, q)
        out.append(is_crc[[r for r in row if r != self_pos][:k]].mean())
    return pd.Series(out, index=latent.index[query.values])


def consensus(pole, crc_likeness, random_state=0):
    """Two-component GMM splits CRC-likeness; both criteria must agree, otherwise 'discordant'."""
    gmm = GaussianMixture(n_components=2, random_state=random_state).fit(crc_likeness.values.reshape(-1, 1))
    high = gmm.predict_proba(crc_likeness.values.reshape(-1, 1))[:, int(np.argmax(gmm.means_.ravel()))] >= 0.5
    group = np.where((pole == "CRC") & high, "CRC-like", np.where((pole == "UC") & ~high, "UC-like", "discordant"))
    return pd.Series(group, index=crc_likeness.index), pd.Series(high, index=crc_likeness.index)
