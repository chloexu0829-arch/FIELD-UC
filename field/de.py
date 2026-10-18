"""Python wrapper around limma_voom.R and the module-selection rule."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

R_SCRIPT = Path(__file__).with_name("limma_voom.R")


def limma(counts, meta, formula, bootstrap_module=None, n_boot=C.BOOT_N, seed=C.BOOT_SEED):
    """counts: genes x samples raw counts; meta: group (UC-like/CRC-like), subject and covariates.
    Returns the full topTable, or bootstrap reselection frequencies if bootstrap_module is given."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        counts.to_csv(tmp / "counts.csv")
        meta.loc[counts.columns].to_csv(tmp / "meta.csv")
        cmd = ["Rscript", str(R_SCRIPT), "fit", str(tmp / "counts.csv"), str(tmp / "meta.csv"), formula, str(tmp / "out.csv")]
        if bootstrap_module is not None:
            bootstrap_module.to_csv(tmp / "module.csv", index=False)
            cmd[2] = "boot"
            cmd += [str(tmp / "module.csv"), str(n_boot), str(seed)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        print(run.stdout.strip())
        return pd.read_csv(tmp / "out.csv")


def select_module(de, fdr=C.DE_FDR, lfc=C.DE_LFC, top=C.DE_TOP):
    """Genes with BH < fdr and |log2FC| > lfc, capped at the top `top` by |log2FC|.
    Direction = sign(log2FC); weight = |log2FC|."""
    de = de.assign(direction=np.sign(de.logFC).astype(int), weight=de.logFC.abs())
    de["in_module"] = de.gene.isin(de[(de["adj.P.Val"] < fdr) & (de.weight > lfc)].nlargest(top, "weight").gene)
    return de
