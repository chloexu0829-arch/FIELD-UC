"""External GEO cohorts: series-matrix parsing, probe-to-symbol collapse and class labels."""
import csv
import gzip

import numpy as np
import pandas as pd

import config as C

# label: where the group label lives; negative/positive: patterns matched exactly, then as substrings
COHORTS = {
    "GSE3629": dict(matrix="GSE3629_series_matrix.txt.gz", gpl="GPL570-55999.txt", label="description",
                    negative=["UC-NonCa"], positive=["UC-Ca", "UC-associated Ca"]),
    "GSE47908": dict(matrix="GSE47908_series_matrix.txt.gz", gpl="GPL570-55999.txt", label="disease state",
                     negative=["control", "left-sided", "pancolitis"], positive=["dysplasia"]),
    "GSE236055": dict(matrix="GSE236055_series_matrix.txt.gz", gpl="GPL21185-21174.txt", label="progressor_status",
                      negative=["NP"], positive=["P"]),
    "GSE220063": dict(matrix="GSE220063_Proj_07182_W_counts_scaled_DESeq.xlsx", label="suffix",
                      negative=["CRC"], positive=["CAC"]),
    "GSE37283": dict(matrix="GSE37283_series_matrix.txt.gz", gpl="GPL13158-5065.txt", label="title",
                     negative=["normal", "quiescent"], positive=["neoplasia"]),
}


def _rows(path, delimiter=None):
    """csv rows of a plain or gzipped GEO text file; tab- or comma-delimited (sniffed from line 1)."""
    with open(path, "rb") as fh:
        opener = gzip.open if fh.read(2) == b"\x1f\x8b" else open
    with opener(path, "rt", encoding="utf-8-sig") as fh:
        if delimiter is None:
            delimiter = "\t" if "\t" in fh.readline() else ","
            fh.seek(0)
        yield from csv.reader(fh, delimiter=delimiter)


def read_series_matrix(path):
    """Return (probes x samples expression, per-sample annotation: title, description, characteristics)."""
    info, table, in_table = {}, [], False
    for row in _rows(path):
        if not row:
            continue
        head = row[0].strip()
        if head in ("!series_matrix_table_begin", "!series_matrix_table_end"):
            in_table = head.endswith("begin")
        elif in_table:
            table.append(row)
        elif head in ("!Sample_title", "!Sample_description") and head[8:] not in info:
            info[head[8:]] = row[1:]
        elif head == "!Sample_characteristics_ch1":
            key = next(v for v in row[1:] if v.strip()).split(":", 1)[0].strip()
            info[key] = [v.split(":", 1)[1].strip() if ":" in v else v.strip() for v in row[1:]]
    samples = [s for s in table[0][1:] if s]
    expr = pd.DataFrame([r[1:len(samples) + 1] for r in table[1:]], index=[r[0].strip() for r in table[1:]],
                        columns=samples).apply(pd.to_numeric, errors="coerce")
    return expr, pd.DataFrame({k: v[:len(samples)] for k, v in info.items()}, index=samples)


def gpl_symbols(path):
    """Probe -> first HGNC symbol from a GEO platform table (blank and '---' entries dropped)."""
    mapping, column = {}, None
    for row in _rows(path, delimiter="\t"):
        if not row or row[0].startswith(("!", "#", "^")):
            continue
        if column is None:
            if row[0].startswith("ID"):
                column = next(i for i, c in enumerate(row)
                              if c.lower().replace("_", " ") in {"gene symbol", "genesymbol", "symbol"})
            continue
        symbol = row[column].split("///")[0].strip() if len(row) > column else ""
        if symbol and symbol != "---":
            mapping[row[0].strip()] = symbol
    return mapping


def collapse_probes(expr, mapping):
    """One row per symbol: the probe with the highest mean intensity in the cohort."""
    symbol = expr.index.map(mapping)
    expr = expr[symbol.notna()].set_axis(symbol[symbol.notna()], axis=0)
    order = np.argsort(-expr.mean(axis=1).to_numpy(), kind="stable")
    expr = expr.iloc[order]
    return expr[~expr.index.duplicated(keep="first")]


def _classify(label, cfg):
    label = str(label).strip().lower()
    for match in (lambda p: label == p.lower(), lambda p: p.lower() in label):
        for cls in ("negative", "positive"):
            if any(match(p) for p in cfg[cls]):
                return cls
    return "other"


def load_cohort(name):
    """Return (genes x samples expression, 0/1 labels, sample annotation) for the analysed samples."""
    cfg, path = COHORTS[name], C.GEO_DIR / COHORTS[name]["matrix"]
    if cfg["label"] == "suffix":   # GSE220063: DESeq-scaled counts, sample names end in __CAC / __CRC
        raw = pd.read_excel(path)
        expr = raw.set_index("GeneSymbol").drop(columns="ID").astype(float).groupby(level=0).max()
        info = pd.DataFrame({"suffix": [c.rsplit("__", 1)[-1] for c in expr.columns]}, index=expr.columns)
    else:
        expr, info = read_series_matrix(path)
        expr = collapse_probes(expr, gpl_symbols(C.GEO_DIR / cfg["gpl"]))
    group = info[cfg["label"]].map(lambda v: _classify(v, cfg))
    keep = group.isin(["negative", "positive"])
    return expr.loc[:, keep], group[keep].eq("positive").astype(int), info.loc[keep]


def patient_level(score, y, patient):
    """Average biopsy scores within patient (GSE236055); a patient is positive if any biopsy is."""
    frame = pd.DataFrame({"score": score, "y": y, "patient": patient})
    agg = frame.groupby("patient").agg(score=("score", "mean"), y=("y", "max"))
    return agg.y, agg.score
