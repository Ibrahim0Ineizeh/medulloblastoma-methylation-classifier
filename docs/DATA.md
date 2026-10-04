# Data guide and provenance

I used methylation data from the public **GSE130051** study for my assessments.
The supporting Python workflow starts from four prepared files used in the second
assessment. It can run a synthetic demo from a fresh checkout, but rerunning the
recorded research benchmark needs those local files. They are not included in
this public repository, and the saved work does not provide a public download
route for the exact prepared derivatives.

## Scientific sources

- [GEO accession GSE130051](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130051): the public repository accession for the underlying samples and array data.
- [Sharma et al. (2019), *Second-generation molecular subgrouping of medulloblastoma*](https://doi.org/10.1007/s00401-019-02020-0): the consensus study underpinning the eight reference subtype labels.
- [Abid and Rafiee (2025), *Cross-Platform DNA Methylation Classifier for the Eight Molecular Subtypes of Group 3 & 4 Medulloblastoma*](https://arxiv.org/abs/2510.02416): the methodological context for the processed representation, original split, and EPIC validation strategy.

The labels are research reference assignments. They are not independently adjudicated diagnoses produced by this project. Subtypes `1` through `8` are **categories**, not an ordered numerical outcome. The broader `MB_G3` and `MB_G4` subgroup labels are metadata rather than the classifier target.

## Required local inputs

If you have the original prepared bundle, place these four files in `data/raw/`:

| File | Prepared contents | Approximate file size |
| --- | --- | --- |
| `hm450_final.parquet` | 1,370 rows; 22,892 CpG probe columns; an embedded `subtype` column | 181.6 MB |
| `epic_final.parquet` | 97 rows; the same 22,892 CpG probe columns; an embedded `subtype` column | 29.6 MB |
| `gse130051_metadata.csv` | 1,501 sample records, including GSM ID, array platform, cohort, subgroup, and subtype | 0.14 MB |
| `Train_Set1_MB_CommonProbes.csv` | 1,016 training IDs, an existing 13,916-probe panel, and a `Group` label column | 254.6 MB |

Sizes use decimal megabytes and are descriptive rather than integrity checks.
The [recorded input manifest](../reports/benchmark_full/input_manifest.json)
contains exact byte sizes and SHA-256 digests. Matching its four entries identifies
the inputs used for the saved full-profile run; the manifest does not provide the
files themselves. The audit command below hashes the inputs you supply.

The `gsm` index in the parquet files and the `Samples` column in the training CSV are the sample identifiers. The metadata CSV may include a saved dataframe index named `Unnamed: 0`; this is not a biological feature. The embedded parquet `subtype` columns and training CSV `Group` column are also **never included in model features**.

In my original local workspace, these files are under `Assessment 2/data/`.
That directory is not part of the public checkout. I can use it directly with
`--data-dir 'Assessment 2/data'`; the documented `data/raw/` layout works without
the assessment folder structure when the prepared bundle is available.

```text
data/
  raw/
    hm450_final.parquet
    epic_final.parquet
    gse130051_metadata.csv
    Train_Set1_MB_CommonProbes.csv
```

The repository keeps the data provenance and input checksums, while the large
prepared files remain outside Git. Public GEO files are upstream sources;
downloading them alone does not supply the four prepared files listed here.

## Run with the prepared inputs

The default installation in [the README](../README.md) is enough to run the
workflow. To match the recorded numerical versions, create an environment using
Python **3.13.7**, then install the lock file before the package:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Run these commands from the repository root after adding the four inputs:

```bash
mb-classifier audit --data-dir data/raw --output reports/my_run/data_audit.json
mb-classifier train --data-dir data/raw --profile full --seed 42 --output reports/my_run
```

Compare the audit's `inputs` entries with the recorded manifest to check that you
have the same files. Training creates `model.joblib`, `metrics.json`,
`metrics.csv`, figures and `index.html` in the output directory, and records its
environment and input hashes. Open `reports/my_run/index.html` locally to read
the report. Regenerate it from an existing run with:

```bash
mb-classifier report --output reports/my_run
```

The [lock file](../requirements-lock.txt) pins numerical libraries, not the
original Colab/R processing environment. Matching files, seed and versions makes
the later prepared-data run comparable; it does not recreate the original
assessment experiments or the upstream raw-data preparation.

## What is reused from each assessment

Assessment 1 contributes the data preparation context: probe quality filtering, beta-value inspection, feature reduction, and exploratory visualizations. Assessment 2 contributes the prepared matrices, explicit sample identifiers, classifier comparison, and validation questions.

The supporting implementation aligns the shared CpG columns in the two prepared
parquet matrices. The original files have matching **22,892-probe** schemas, so
all those CpGs form its starting feature universe. It reads the training CSV to
recover **training sample IDs**, and checks its `Group` labels against metadata.
It does not reuse that CSV's selected 13,916-probe panel or either assessment's
fitted NMF representation.

This distinction matters because the old workflow correlated beta values with numeric subtype codes. That approach assumes an ordering of nominal classes. The current pipeline selects probes by training-fold variance and treats all eight targets as categories.

The prepared 22,892-probe universe is still inherited. The complete sequence of
filters that produced it, and whether those choices were independent of every
historical evaluation sample, cannot be established from the supplied files
alone. Fitting the later pipeline inside cross-validation addresses its own
learned transformations; it cannot undo an undocumented earlier selection step.

## Cohort and split definitions

The prepared `hm450_final.parquet` matrix contains 1,271 annotated samples whose recorded array platform is `GPL13534` and 99 annotated samples whose recorded platform is `GPL21145`. Its filename describes the processed feature representation and must not be interpreted as proof that every row was physically assayed on a 450K array.

| Evaluation role | Number of rows | Definition |
| --- | --- | --- |
| Training | 1,016 | IDs in the supplied training CSV, restricted to the annotated 450K-platform cohort |
| Internal test | 255 | Remaining annotated 450K-platform IDs after reserving training |
| Published processed representation of the EPIC cohort | 99 | All annotated EPIC-platform metadata IDs present in the primary parquet matrix |
| Independently processed EPIC representation | 97 | Available rows in `epic_final.parquet`, all from the reserved EPIC cohort |

**All 99 annotated EPIC IDs are excluded from training and the internal test.** Exclusion is based on metadata, not just the 97 rows that happen to be present in the second matrix. Otherwise the two missing EPIC rows could be incorrectly assigned to the internal test.

The 97 EPIC rows have IDs that also occur in the 99-row processed representation. They describe the same sample records through different processed representations, so these evaluation sets are **paired**, not independent patient cohorts. A concordance analysis therefore uses their common 97 IDs in the same order. It does not add their sample counts together to claim 196 independent validation cases.

The labeled EPIC IDs absent from `epic_final.parquet` are `GSM3877833` and `GSM3877848`. The cross-platform paper reports excluding two samples for excessive missingness. This is context for the historical preparation; the supporting loader works with the 97 complete prepared rows and does not reconstruct that exclusion from raw intensities.

## Input checks

The data audit verifies the properties needed for a meaningful experiment:

- Sample and feature names are unique, and every selected sample has usable metadata.
- Targets are limited to the eight expected categories and aligned by sample ID.
- Features are CpG probe columns; saved indices, labels, and metadata cannot enter the feature matrix.
- Beta values are finite and lie between zero and one.
- The loader takes the intersection of CpG columns across the two parquet
  inputs, excludes nonshared probes and metadata columns, and puts the shared
  probes in a consistent order. The inputs do not have to start with identical
  schemas; the recorded files each have 22,892 CpGs, all shared, with zero
  CpGs excluded during alignment.
- Training, internal test, and reserved EPIC-cohort IDs are disjoint.
- EPIC rows are a subset of the reserved cohort, and paired comparisons align common IDs explicitly.

The recorded matrices have no missing CpG values, unique sample IDs and probe
names, and values within the expected beta range. Their embedded labels agree
with metadata, as does the training CSV's `Group` column. A fresh audit checks
the aligned inputs rather than relying on these observations. With other input
schemas, inspect the shared and excluded probe counts before interpreting a run.

## Prediction table format

`mb-classifier predict` expects a CSV with one row per specimen, a `sample_id`
column, and the original input CpG columns recorded in the model bundle. Use
`--id-column` to supply a different identifier column. Column order may differ;
the saved schema restores training order. Missing required probes, duplicate
sample IDs, nonfinite values, and values outside `[0,1]` cause a clear error.
Additional columns are ignored after selecting the saved schema.

For a CSV you supply at `data/raw/prediction_profiles.csv`, predict with the
model saved by the run above:

```bash
mb-classifier predict --model reports/my_run/model.joblib \
  --input data/raw/prediction_profiles.csv \
  --output reports/my_run/predictions/new_profiles.csv
```

This command needs your prediction table; it is not another included dataset.
The model bundle is generated locally during training and is also excluded
from the public repository.

The saved pipeline includes the variance selector, so prediction currently
requires the full training input schema rather than just its retained probes.
The prediction command writes specimen IDs and `predicted_subtype`; it does not
output calibrated diagnostic probabilities. Use model bundles created by your
own trusted runs: joblib serialization can execute code during loading.

## Reproducibility boundary

The [study notebook](../notebooks/medulloblastoma_study.ipynb) records my original
preparation steps: parsing the two GEO series-matrix files, extracting EPIC beta
values with R/SeSAMe, applying probe masks and filtering, and saving CSV matrices.
The second assessment then loads the prepared parquet files and training panel.
It does not include the complete process that generated those exact inputs.

GEO provides the underlying study data, and the notebook records the raw filenames
and preparation I used. Rebuilding the exact four-file bundle would also require
the missing preparation steps, annotations and original environment. There is
therefore no verified raw-data command here that reproduces the recorded input
hashes. The executable package covers modeling **from prepared beta matrices**;
the synthetic demo remains the reproducible option requiring no research files.

The public study notebook preserves the original analysis with assessment covers
and student numbers removed. Marking feedback, assessment briefs, presentation
videos, and third-party example solutions remain outside the public repository.
