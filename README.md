# Medulloblastoma classification from DNA methylation

**Ibrahim Ineizeh · AI for Health project**

I investigated how DNA methylation profiles can distinguish the eight molecular
subtypes of Group 3 and Group 4 medulloblastoma. I started with data preparation
and exploratory analysis, then developed an SVM classifier, investigated its
errors, and compared results on HM450-compatible and EPIC data.

The project combines my two assessments into one study. My original explanations,
code, figures and results are the main material here.

**Start with [the project report](REPORT.md)** for the full analysis, or browse
[the combined research notebook](notebooks/medulloblastoma_study.ipynb) for my
original code and recorded outputs.

## What I worked on

- Prepared methylation data and GEO metadata, removed unsuitable probes, and
  aligned the features available in both representations. The original HM450
  beta-value file was approximately 13 GB, so I used Colab Pro to work with it.
- Explored methylation patterns with NMF and visualised the resulting component
  scores using PCA.
- Trained an SVM, examined class imbalance and individual errors, and used
  oversampling and grid search to improve the classifier.
- Increased NMF from 10 to 15 components after reviewing the remaining test
  errors, then evaluated the model on the available validation representations.

## My original results

These are the results recorded in my submitted analysis, before the later
software refactoring:

| Experiment | Evaluation samples | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Baseline SVM, 10 NMF components | 255 | 95.69% | 0.9472 |
| Tuned SVM, 10 NMF components | 255 | 96.08% | 0.9569 |
| Tuned SVM, 15 NMF components | 255 | **97.65%** | **0.9665** |
| Final model, HM450-compatible validation | 99 | 93.94% | 0.9151 |
| Final model, EPIC validation | 97 | 92.78% | 0.9045 |

![Original test-set confusion matrix for the 15-component model](figures/nmf15-confusion.png)

The final model reduced test errors from 11 in the baseline to 6. Subtype 3/4
confusion remained an important part of my error analysis. The validation sets
share specimen records; they should be read as paired representations rather
than two independent patient cohorts. The report discusses these findings and
the limits of the original evaluation.

## Run the supporting Python workflow

The `src/` package was added when preparing the study for GitHub. It provides
data checks, training, prediction and report generation. Its NMF and probe
selection are fitted inside cross-validation. This is a later implementation,
and its results are kept separate from my original study.

From the repository root, a fresh checkout can run the synthetic demo and tests
without downloading research data:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
mb-classifier demo --output reports/demo
python -m pytest -q
```

Open `reports/demo/index.html` to see the generated report. The demo checks that
the software runs; its scores are not the research results above.

The research rerun requires four prepared files listed in [the data guide](docs/DATA.md).
They are retained locally but **are not bundled with this public repository**.
The GEO accession provides the upstream data, not a download for these exact
prepared derivatives. If you have those files, place them in `data/raw/`, then run:

```bash
mb-classifier audit --data-dir data/raw --output reports/my_run/data_audit.json
mb-classifier train --data-dir data/raw --profile full --seed 42 --output reports/my_run
```

Training saves a model, aggregate metrics and an HTML report. The data guide also
gives the pinned-environment setup, recorded input checksums, and the command for
[predicting new profiles](docs/DATA.md#prediction-table-format).

The [executable walkthrough](notebooks/reproduce_pipeline.ipynb) covers this
supporting workflow. One [recorded packaged run](reports/benchmark_full/index.html)
is retained for comparison; download the repository and open its HTML locally.
That later run used 15-component NMF followed by a linear-kernel SVC and obtained
98.04% internal-test accuracy. It is **not** the source of the original results
above. [Method notes](docs/METHODOLOGY.md) explain the differences and reproduction limits.

## Files to read

| File or folder | Contents |
|---|---|
| [REPORT.md](REPORT.md) | Combined study, based on my original writing |
| [notebooks/medulloblastoma_study.ipynb](notebooks/medulloblastoma_study.ipynb) | Original code, explanations and saved results |
| `figures/` | Study figures; baseline confusion-matrix title and axes corrected, with original counts preserved |
| `src/` and `tests/` | Supporting Python implementation and checks |
| `docs/` | Data requirements and concise method notes |
| `reports/benchmark_full/` | Recorded later experiment; rerunning it requires the prepared inputs |

## Sources

The data and subtype definitions come from [GEO GSE130051](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130051)
and [Sharma et al. (2019)](https://doi.org/10.1007/s00401-019-02020-0).
The cross-platform methods were informed by
[Abid and Rafiee (2025)](https://arxiv.org/abs/2510.02416).

Original project code is covered by the [MIT license](LICENSE); source datasets
and publications retain their own terms. The notebook retains my original
disclosure of assistance with plotting, tables and proofreading. The later
packaging, report consolidation and evaluation fixes were added with Codex.
