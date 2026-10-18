"""Discovery data: IBDome, IBDMDB and TCGA counts, UC biopsy metadata, Patient-M input matrix."""
import sqlite3

import numpy as np
import pandas as pd
from combat.pycombat import pycombat
from sklearn.decomposition import PCA

import config as C

COLORECTAL_SITES = {
    "rectum", "sigmoid colon", "colon sigmoid", "descending (left-sided) colon", "colon descending",
    "left colon", "transverse colon", "colon transverse", "ascending (right-sided) colon",
    "colon ascending", "right colon", "cecum", "caecum", "colon",
}


def _sql(query):
    con = sqlite3.connect(C.IBDOME_SQLITE)
    try:
        return pd.read_sql(query, con)
    finally:
        con.close()


def ibdome_counts(agg="sum", round_counts=False):
    """IBDome raw counts (HGNC symbols x samples); duplicated symbols summed or averaged."""
    raw = pd.read_csv(C.IBDOME_COUNTS, index_col=0)
    if round_counts:
        raw = raw.round(0)
    genes = _sql("SELECT ensg, hgnc_symbol FROM genes").dropna()
    symbol = raw.index.map(dict(zip(genes.ensg, genes.hgnc_symbol)))
    keep = symbol.notna()
    return raw[keep].set_axis(symbol[keep], axis=0).groupby(level=0).agg(agg)


def ibdmdb_counts(hgnc_only=False):
    """IBDMDB host-transcriptomics raw counts (symbols x samples); duplicated symbols summed."""
    raw = pd.read_csv(C.IBDMDB_COUNTS, index_col=0, skiprows=1)
    raw.index = raw.index.astype(str)
    if hgnc_only:
        raw = raw[raw.index.str.match(r"^[A-Za-z][A-Za-z0-9_.-]*$")]
    return raw.groupby(level=0).sum()


def uc_counts(sample_ids):
    """Raw counts for UC biopsies on the IBDome-IBDMDB shared gene universe used by every DE model."""
    ibo, ibd = ibdome_counts(), ibdmdb_counts()
    genes = ibo.index.intersection(ibd.index)
    counts = pd.concat([ibo.loc[genes], ibd.loc[genes]], axis=1)
    return counts[[s for s in sample_ids if s in counts.columns]]


def uc_biopsy_meta():
    """One row per UC RNA-seq biopsy: cohort, subject, inflammation, site, same-site histology."""
    ibo = _sql("""
        WITH h AS (
            SELECT hs.subject_id, hs.date, hs.tissue, AVG(h.normalized_riley_score) AS local_histology
            FROM data_histopathology2 h JOIN samples_subjects_tissues hs ON h.sample_id = hs.sample_id
            WHERE h.normalized_riley_score IS NOT NULL OR h.normalized_naini_cortina_score IS NOT NULL
            GROUP BY hs.subject_id, hs.date, hs.tissue)
        SELECT s.sample_id, s.subject_id, s.tissue, i.inflammation_status AS inflammation, h.local_histology
        FROM samples_subjects_tissues s
        LEFT JOIN samples_subjects_tissues_with_inflammation i ON s.sample_id = i.sample_id
        LEFT JOIN h ON s.subject_id = h.subject_id AND s.date = h.date AND s.tissue = h.tissue
        WHERE s.sample_type = 'RNA-seq' AND s.disease = 'Ulcerative colitis'""")
    hmp = pd.read_csv(C.HMP2_META, low_memory=False)
    hmp = hmp[hmp.data_type.eq("host_transcriptomics")]
    ibd = pd.DataFrame({
        "sample_id": hmp["External ID"].astype(str),
        "subject_id": hmp["Participant ID"].astype(str),
        "tissue": hmp["biopsy_location"],
        "inflammation": hmp["is_inflamed"].map({"Yes": "inflamed", "No": "non_inflamed"}),
        "local_histology": np.nan,
    })
    meta = pd.concat([ibo.assign(cohort="IBDome"), ibd.assign(cohort="IBDMDB")])
    meta = meta.drop_duplicates("sample_id").set_index("sample_id")
    meta["is_colorectal"] = meta.tissue.map(
        lambda t: isinstance(t, str) and " ".join(t.lower().split()) in COLORECTAL_SITES)
    return meta


# ---------------------------------------------------------------- Patient-M input
def logcpm(counts):  # samples x genes
    return np.log2(counts.div(counts.sum(axis=1).replace(0, 1), axis=0) * 1e6 + 1)


def _zscore(df):  # per gene, population SD
    v = df.values.astype(np.float64)
    sd = v.std(axis=0, keepdims=True)
    return pd.DataFrame((v - v.mean(axis=0, keepdims=True)) / np.where(sd < 1e-8, 1.0, sd),
                        index=df.index, columns=df.columns)


def _tcga(project):
    counts_path, clinical_path = C.TCGA[project]
    raw = pd.read_csv(counts_path, index_col=0)
    histology = pd.read_csv(clinical_path).set_index("bcr_patient_barcode")["histological_type"]
    tumour = [c for c in raw.columns
              if c[13:15] == "01" and histology.get(c[:12], "Unknown") != "Unknown"]
    normal = [c for c in raw.columns if c[13:15] == "11"]
    status = ["CRC_CMA" if "mucinous" in str(histology[c[:12]]).lower() else "CRC_CA" for c in tumour]
    return raw[tumour + normal].T, status + ["CRC_control"] * len(normal)


def representation_matrix():
    """Samples x genes input of Patient-M (991 x 13,477; 988 after outliers) and its annotation.

    Per-cohort log2(CPM+1), prevalence filter (log2CPM > 1 in >= 10% of samples) and z-score;
    four-dataset gene intersection; three-stage ComBat (IBDome vs IBDMDB, COAD vs READ, IBD vs
    TCGA) with a lesion-vs-normal covariate; per-gene z-score. Outliers are flagged, not removed.
    """
    blocks, annotation = [], []

    disease = _sql("SELECT sample_id, disease FROM samples_subjects_tissues WHERE sample_type = 'RNA-seq'")
    ibo = ibdome_counts(agg="mean", round_counts=True)
    ibo_ids = {status: [s for s in disease.sample_id[disease.disease == label] if s in ibo.columns]
               for status, label in [("UC_disease", "Ulcerative colitis"), ("UC_control", "non-IBD")]}
    blocks.append(("IBDome", ibo[ibo_ids["UC_disease"] + ibo_ids["UC_control"]].T,
                   ["UC_disease"] * len(ibo_ids["UC_disease"]) + ["UC_control"] * len(ibo_ids["UC_control"])))

    hmp = pd.read_csv(C.HMP2_META, low_memory=False)
    hmp = hmp[hmp.data_type.eq("host_transcriptomics")]
    ibd = ibdmdb_counts(hgnc_only=True)
    ibd_ids = {status: [s for s in hmp.loc[hmp.diagnosis == label, "External ID"].dropna() if s in ibd.columns]
               for status, label in [("UC_disease", "UC"), ("UC_control", "nonIBD")]}
    blocks.append(("IBDMDB", ibd[ibd_ids["UC_disease"] + ibd_ids["UC_control"]].T,
                   ["UC_disease"] * len(ibd_ids["UC_disease"]) + ["UC_control"] * len(ibd_ids["UC_control"])))

    for project in C.TCGA:
        blocks.append((project, *_tcga(project)))

    scaled = []
    for cohort, counts, status in blocks:
        expr = logcpm(counts)
        scaled.append(_zscore(expr.loc[:, (expr > 1.0).mean(axis=0) >= 0.10]))
        annotation.append(pd.DataFrame({"cohort": cohort, "disease_status": status}, index=counts.index))
    genes = sorted(set.intersection(*(set(s.columns) for s in scaled)))
    meta = pd.concat(annotation)
    lesion = meta.disease_status.isin(["UC_disease", "CRC_CA", "CRC_CMA"]).astype(int)

    def combat(frames, batch):
        x = pd.concat([f[genes] for f in frames])
        return pycombat(x.T, pd.Series(batch), mod=lesion.loc[x.index].tolist()).T

    ibd_part = combat(scaled[:2], ["IBDome"] * len(scaled[0]) + ["IBDMDB"] * len(scaled[1]))
    crc_part = combat(scaled[2:], ["COAD"] * len(scaled[2]) + ["READ"] * len(scaled[3]))
    X = _zscore(combat([ibd_part, crc_part], [0] * len(ibd_part) + [1] * len(crc_part)))

    pcs = PCA(n_components=2, random_state=C.SEED).fit_transform(X.values)
    # outlier rule applied in the analysis (3 samples, each > 6 SD from the PC mean)
    meta["outlier"] = (pcs[:, 0] < -300) | (pcs[:, 1] < -250)
    meta["tissue_state"] = meta.disease_status.map({
        "UC_disease": "inflamed", "UC_control": "healthy", "CRC_CA": "tumor", "CRC_CMA": "tumor",
        "CRC_control": "adjacent_normal"})
    return X, meta
