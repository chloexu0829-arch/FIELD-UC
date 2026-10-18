"""Step 4. Adjusted (FIELD) design: IBDome colorectal biopsies with same-day, same-site histology.

expression ~ group + local_histology + tissue_site (patient blocking); the top-150 rule is repeated in
200 patient-clustered bootstraps and genes reselected in >= 70% form the frozen 47-gene signature.
Outputs: results/adjusted_de.csv, results/field47_signature.csv
"""
import pandas as pd

import config as C
from field.data import uc_counts
from field.de import limma, select_module

poles = pd.read_csv(C.RESULTS / "poles.csv", index_col=0)
design = poles[poles.group.isin(["CRC-like", "UC-like"]) & poles.cohort.eq("IBDome")
               & poles.is_colorectal & poles.local_histology.notna()]
counts = uc_counts(design.index)
meta = design.loc[counts.columns, ["group", "subject_id", "local_histology", "tissue"]]
meta = meta.rename(columns={"subject_id": "subject", "tissue": "tissue_site"})
print(f"{counts.shape[1]} biopsies, {meta.subject.nunique()} patients:", meta.group.value_counts().to_dict())

formula = "~ group + local_histology + tissue_site"
de = select_module(limma(counts, meta, formula))
boot = limma(counts, meta, formula, bootstrap_module=de[["gene", "direction"]])
de = de.merge(boot, on="gene", how="left")
de.to_csv(C.RESULTS / "adjusted_de.csv", index=False)

top = de[de.in_module]
for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
    kept = top[top.selection_frequency >= threshold]
    print(f"reselection >= {threshold:.1f}: {len(kept)} genes ({(kept.direction > 0).sum()} up)")
signature = top[top.selection_frequency >= C.RESELECTION]
signature[["gene", "direction", "weight", "logFC", "adj.P.Val", "selection_frequency", "direction_consistency"]].to_csv(
    C.RESULTS / "field47_signature.csv", index=False)
print(f"frozen signature: {len(signature)} genes ({(signature.direction > 0).sum()} up, "
      f"{(signature.direction < 0).sum()} down)")
