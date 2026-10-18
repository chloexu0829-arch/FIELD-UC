"""File locations and fixed analysis constants (see data/README.md for downloads)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, RESOURCES = ROOT / "data", ROOT / "results", ROOT / "resources"

IBDOME_SQLITE = DATA / "ibdome_v1.0.1.sqlite"
IBDOME_COUNTS = DATA / "IBDome_transcriptomics_rawcounts_v1.0.1.csv"
IBDMDB_COUNTS = DATA / "ibdmdb_expression_counts.csv"
HMP2_META = DATA / "hmp2_metadata_2018-08-20.csv"
TCGA = {  # project: (raw counts, clinical patient table)
    "TCGA-COAD": (DATA / "TCGA_COAD_counts.csv", DATA / "TCGA_COAD_clinical.csv"),
    "TCGA-READ": (DATA / "TCGA_READ_counts.csv", DATA / "TCGA_READ_clinical.csv"),
}
GEO_DIR = DATA / "geo"

OMNIPATH_GENES = RESOURCES / "omnipath_genes.txt"
HALLMARK_INFLAMMATION = RESOURCES / "hallmark_inflammatory_response.txt"

SEED = 42                      # representation and clustering
DE_FDR, DE_LFC, DE_TOP = 0.05, 1.0, 150
BOOT_N, BOOT_SEED, RESELECTION = 200, 20260804, 0.70
NULL_N, NULL_SEED = 1000, 20260809
MIN_COVERAGE = 0.80            # minimum fraction of module genes present to score a cohort
