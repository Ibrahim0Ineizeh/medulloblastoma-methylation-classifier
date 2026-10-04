# Original study and later implementation

The [report](../REPORT.md) presents my original study. Its performance values and
confusion counts come from my assessment notebooks. The baseline confusion-matrix
title and axes were corrected while preserving its counts. The
[combined notebook](../notebooks/medulloblastoma_study.ipynb) retains my
explanations, code and recorded results, with clearly marked consolidation notes.

## Original experiments

In the first assessment, I prepared methylation profiles, used train-only variability
and correlation filtering to produce 9,010 shared CpG probes, and explored
NMF/PCA patterns. Its NMF was fitted to a transposed matrix for exploration.

In the second assessment, I used different prepared inputs. I intersected a supplied
13,916-probe training panel with the prepared matrices to obtain 11,234 probes,
used conventional samples-by-probes NMF, and trained SVM classifiers. I compared
a balanced baseline with oversampling and grid search, then increased NMF from
10 to 15 components after examining test errors.

Recorded final results were 97.65% accuracy and 0.9665 macro F1 on 255 test samples,
93.94% accuracy on 99 HM450-compatible validation rows, and 92.78% accuracy on
97 EPIC rows. These original numbers are not replaced by the later rerun.

## What was changed during packaging

The reusable Python workflow starts from the 22,892 CpGs in the prepared parquet
files. It retains the supplied training specimen IDs and discards the old
correlation-selected panel. A pipeline fits variance selection, NMF and scaling
inside each CV fold, then fits a class-weighted SVC with a linear or RBF kernel.
A direct `LinearSVC` on selected probes and a majority-class dummy provide
comparisons. Training CV macro F1 selects the model.

| Profile | Selected probes | CV folds | NMF ranks |
|---|---:|---:|---|
| Quick | 1,000 | 3 | 10, 15 |
| Full | 5,000 | 5 | 10, 15 |

One [full-profile run](../reports/benchmark_full/metrics.csv) is retained with
its environment, input hashes and report. The winner was 15-component NMF
followed by `SVC(kernel="linear", C=0.1)`, rather than the direct `LinearSVC`
comparison. It obtained 98.04% accuracy / 0.9722 macro F1 on the same 255 test records.
This is a separate later experiment with a different feature-selection and
imbalance strategy, not a relabeling of the original assessment result.

A fresh checkout can execute the synthetic workflow. Rerunning this research
experiment requires four prepared files that are not publicly bundled; the
[data guide](DATA.md) gives their schemas, input checksums, pinned environment,
and audit, training and prediction commands. The public GEO data are the source
of the study, but the notebook does not reconstruct the exact prepared bundle.

## Limits that affect interpretation

- The original NMF was fitted before SVM cross-validation, and the rank change
  followed test inspection. Those historical scores describe exploratory work.
  The later pipeline fixes its own fold-fitting boundary but uses the same
  already inspected cohort.
- The starting 22,892-probe matrices are inherited prepared artifacts. Their
  complete upstream curation cannot be independently reconstructed here.
- The 99 HM450-compatible and 97 EPIC validation rows overlap on 97 specimen IDs.
  They are paired processing representations, not independent validation cohorts.
- The original “decision margin” is a top-two probability gap. A confident error
  does not prove that a reference subtype is wrong, and good array performance
  does not prove absence of batch effects or a validated biological mechanism.

The original notebook is a preserved research record. Its raw-data steps require
additional GEO files, masks, Colab dependencies and the R/SeSAMe environment.
It also contains a saved consensus plot whose `ks` and `scores` definitions are
missing from the submitted code. These boundaries are marked beside the relevant
cells. The supporting [execution notebook](../notebooks/reproduce_pipeline.ipynb)
and Python package can run with synthetic data without those original files.
