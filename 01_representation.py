"""Step 1. Patient-M representation of 988 transcriptomes and K-medoids clustering.

Output: results/patient_latent_codes.csv (32-d representation, cluster, cohort, disease status).
Retraining on other hardware gives a numerically different latent space. Pass --weights to
reuse an existing Patient-M state dict instead of training.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

import config as C
from field.data import representation_matrix
from field.patient_m import PatientM, encode, train
from field.poles import kmedoids, select_k

parser = argparse.ArgumentParser()
parser.add_argument("--weights", help="released Patient-M state dict (skips training)")
args = parser.parse_args()
torch.manual_seed(C.SEED)
np.random.seed(C.SEED)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

X, meta = representation_matrix()
cumvar = np.cumsum(PCA(n_components=30, random_state=C.SEED).fit(X.values).explained_variance_ratio_)
target_rank = min(int(np.searchsorted(cumvar, 0.45 if np.searchsorted(cumvar, 0.70) + 1 > 15 else 0.70)) + 1, 28)
X, meta = X[~meta.outlier], meta[~meta.outlier]
genes = sorted(set(C.OMNIPATH_GENES.read_text().split()) & set(X.columns))   # interaction-network genes
x = X[genes].to_numpy(np.float32)
print(f"{len(X)} samples x {X.shape[1]} genes; Patient-M input: {len(genes)} genes")
print(meta.groupby(["cohort", "disease_status"]).size().to_string())

if args.weights:
    model = PatientM(len(genes)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
else:
    model = train(x, meta.cohort.values, target_rank=target_rank, seed=C.SEED, device=device)
    torch.save(model.state_dict(), C.RESULTS / "patient_m.pt")
Z = encode(model, x, device)

K, k_table = select_k(Z, seed=C.SEED)
print(k_table.round(3).to_string(index=False), f"\nselected K = {K}")
labels, medoids = kmedoids(Z, K, random_state=C.SEED)

out = pd.DataFrame(Z, index=X.index, columns=[f"z_{i}" for i in range(Z.shape[1])])
out = out.join(meta[["cohort", "disease_status", "tissue_state"]]).assign(cluster=labels, is_medoid=False)
out.iloc[medoids, out.columns.get_loc("is_medoid")] = True
out.to_csv(C.RESULTS / "patient_latent_codes.csv")
k_table.to_csv(C.RESULTS / "k_selection.csv", index=False)
