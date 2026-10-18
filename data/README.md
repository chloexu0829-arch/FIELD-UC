# Data

Place the following public files in this folder, using the names shown. Nothing here is committed.

## Discovery cohorts

| File | Source |
| --- | --- |
| `ibdome_v1.0.1.sqlite` | [IBDome v1.0.1 database](https://zenodo.org/records/18153919). The tables used are `genes`, `samples_subjects_tissues`, `samples_subjects_tissues_with_inflammation` and `data_histopathology2`. |
| `IBDome_transcriptomics_rawcounts_v1.0.1.csv` | IBDome v1.0.1 RNA-seq raw counts (Ensembl IDs x samples) from [IBDome downloads](https://ibdome.org/#!/data_download) |
| `ibdmdb_expression_counts.csv` | IBDMDB/HMP2 host-transcriptomics raw counts from [IBDMDB results](https://ibdmdb.org/results), saved as CSV with gene symbols x External IDs and one line above the header |
| `hmp2_metadata_2018-08-20.csv` | IBDMDB/HMP2 sample metadata from [IBDMDB results](https://ibdmdb.org/results) |
| `TCGA_COAD_counts.csv`, `TCGA_READ_counts.csv` | TCGA open-access STAR gene counts from the [GDC](https://portal.gdc.cancer.gov) (e.g. via TCGAbiolinks): gene symbols x aliquot barcodes |
| `TCGA_COAD_clinical.csv`, `TCGA_READ_clinical.csv` | TCGA BCR clinical patient tables with `bcr_patient_barcode` and `histological_type` |

## External cohorts (`data/geo/`)

From [GEO](https://www.ncbi.nlm.nih.gov/geo/). Series matrices may be gzipped or plain text.

| File | Cohort |
| --- | --- |
| `GSE3629_series_matrix.txt.gz` | UC with and without cancer (GPL570); current loader includes `UC-Ca` (10) and `UC-associated Ca` (6) as positive, versus `UC-NonCa` (43) |
| `GSE47908_series_matrix.txt.gz` | UC, dysplasia and controls (GPL570) |
| `GSE236055_series_matrix.txt.gz` | progressors vs non-progressors (GPL21185) |
| `GSE37283_series_matrix.txt.gz` | UC with remote neoplasia vs quiescent UC (GPL13158) |
| `GSE220063_Proj_07182_W_counts_scaled_DESeq.xlsx` | CAC vs sporadic CRC, series supplementary file |
| `GPL570-55999.txt`, `GPL13158-5065.txt`, `GPL21185-21174.txt` | platform annotation tables ("Download full table") |
