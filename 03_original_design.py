"""Step 3. Original design: pooled IBDome + IBDMDB, all biopsy sites, no activity adjustment.

expression ~ group + cohort (patient blocking); top 150 genes by |log2FC| among BH < 0.05, |log2FC| > 1.
Outputs: results/original_de.csv (all genes tested), results/original_module.csv (150 genes)
"""
import pandas as pd

import config as C
from field.data import uc_counts
from field.de import limma, select_module

poles = pd.read_csv(C.RESULTS / "poles.csv", index_col=0)
# definite poles with a recorded inflammation status (121 biopsies, 87 patients)
design = poles[poles.group.isin(["CRC-like", "UC-like"]) & poles.inflammation.notna()]
counts = uc_counts(design.index)
meta = design.loc[counts.columns, ["group", "cohort", "subject_id"]].rename(columns={"subject_id": "subject"})
print(f"{counts.shape[1]} biopsies, {meta.subject.nunique()} patients:", meta.group.value_counts().to_dict())

de = select_module(limma(counts, meta, "~ group + cohort"))
de.to_csv(C.RESULTS / "original_de.csv", index=False)
module = de[de.in_module]
module[["gene", "direction", "weight", "logFC", "adj.P.Val"]].to_csv(C.RESULTS / "original_module.csv", index=False)
print(f"original module: {len(module)} genes ({(module.direction > 0).sum()} up, {(module.direction < 0).sum()} down)")
