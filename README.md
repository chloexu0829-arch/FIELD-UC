# FIELD-UC

Code for *Adjusting for histological activity and biopsy site reveals a candidate carcinoma-proximal epithelial state in non-neoplastic ulcerative colitis mucosa*.

FIELD compares an unadjusted UC gene module with a 47-gene signature derived after adjustment for local histological activity and biopsy site. The Patient-M representation is adapted from [GeSubNet](https://github.com/chenzRG/GeSubNet).

## Setup

Python 3.12–3.13 and R 4.4 with limma, edgeR and statmod.

```bash
pip install -r requirements.txt
Rscript -e 'install.packages("BiocManager"); BiocManager::install(c("limma", "edgeR", "statmod"))'
```

## Data

Download the public cohorts listed in [data/README.md](data/README.md). Raw expression data and patient metadata are not included. The two frozen gene signatures are in `results/`; required gene lists are in `resources/`.

## Analysis

Run from the repository root. Scripts 1–4 rebuild discovery; scripts 5–6 evaluate the signatures.

| Script | Analysis |
| --- | --- |
| `01_representation.py` | Patient-M VQ-VAE, K selection and K-medoids clustering |
| `02_poles.py` | Consensus CRC-like and UC-like poles from clusters and kNN CRC-likeness |
| `03_original_design.py` | Pooled, all-site module: `~ group + cohort` |
| `04_adjusted_design.py` | Colorectal module: `~ group + local_histology + tissue_site`; 200 patient bootstraps |
| `05_external_validation.py` | Frozen rank scores, five GEO cohorts and matched random-gene-set nulls |
| `06_progressor_models.py` | Patient-level GSE236055 models and inflammation-stratified comparisons |

```bash
python 01_representation.py
python 02_poles.py
python 03_original_design.py
python 04_adjusted_design.py
python 05_external_validation.py
python 06_progressor_models.py
```

Paths and fixed thresholds are in `config.py`. Shared functions are in `field/`; differential expression uses `field/limma_voom.R`. Generated files are written to `results/`.

To evaluate the supplied signatures directly, download only the external GEO data and run:

```bash
python 05_external_validation.py --scores-only
python 06_progressor_models.py
```

The matched null additionally needs discovery counts and the outputs of scripts 2–4. Rebuilding discovery replaces the supplied signature tables; use a separate checkout to retain them. Retraining Patient-M can change the latent space and pole assignments, so a fresh discovery run is not guaranteed to recover the manuscript values. An existing model can be reused with `python 01_representation.py --weights path/to/patient_m.pt`.

GSE236055 scores and clinical covariates are averaged within patient. This small cross-sectional cohort measures concurrent neoplasia status; the models do not establish prediction of future cancer.

## Reference

Yang Z, Chen Z, et al. GeSubNet: Gene Interaction Inference for Disease Subtype Network Generation. ICLR 2025. [Paper and original implementation](https://github.com/chenzRG/GeSubNet).
