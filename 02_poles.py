"""Step 2. Assign the 198 UC biopsies to CRC-like / UC-like / discordant (shared by both designs).

Output: results/poles.csv
"""
import pandas as pd

import config as C
from field.data import uc_biopsy_meta
from field.poles import cluster_poles, consensus, knn_crc_likeness

latent = pd.read_csv(C.RESULTS / "patient_latent_codes.csv", index_col=0)
z_cols = [c for c in latent.columns if c.startswith("z_")]
uc = latent.disease_status.eq("UC_disease")
crc = latent.disease_status.isin(["CRC_CA", "CRC_CMA"])

frac, pole = cluster_poles(latent)
print("neoplastic fraction per cluster:", frac.round(2).to_dict())
likeness = knn_crc_likeness(latent, z_cols, uc, crc, k=20)
for k in (15, 25, 30):
    rho = knn_crc_likeness(latent, z_cols, uc, crc, k=k).corr(likeness, method="spearman")
    print(f"kNN CRC-likeness, k = {k} vs 20: Spearman rho = {rho:.3f}")
group, high = consensus(pole[uc], likeness)

poles = latent.loc[uc, ["cohort", "cluster"]].assign(pole=pole[uc], crc_likeness=likeness, high_crc_likeness=high,
                                                     group=group)
poles = poles.join(uc_biopsy_meta().drop(columns="cohort"))
poles.to_csv(C.RESULTS / "poles.csv")
print(poles.group.value_counts().to_string())
