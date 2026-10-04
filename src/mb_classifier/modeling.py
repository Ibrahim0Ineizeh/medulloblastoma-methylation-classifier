"""Select models on training folds, then evaluate each held-out cohort once."""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits

from mb_classifier import __version__
from mb_classifier.data import LABELS, validate_dataset
from mb_classifier.features import TopVarianceSelector


def accuracy_interval(correct, total):
    """Wilson 95% binomial interval; uncertainty for accuracy, not macro F1."""
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def evaluate(y_true, y_pred):
    return {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        ),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "accuracy_ci95": accuracy_interval(int(np.sum(np.asarray(y_true) == y_pred)), len(y_true)),
    }


def build_searches(profile, seed, cache_dir, demo=False):
    if profile not in ("quick", "full"):
        raise ValueError("profile must be quick or full")
    n_features = 96 if demo else (1000 if profile == "quick" else 5000)
    n_folds = 2 if demo else (3 if profile == "quick" else 5)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    memory = joblib.Memory(cache_dir, verbose=0)
    nmf = Pipeline(
        [
            ("probes", TopVarianceSelector(n_features)),
            ("nmf", NMF(init="nndsvda", max_iter=4000, tol=1e-3, random_state=seed)),
            ("scaler", StandardScaler()),
            ("classifier", SVC(class_weight="balanced", random_state=seed)),
        ],
        memory=memory,
    )
    ranks = [4, 8] if demo else [10, 15]
    costs = [0.1, 1] if profile == "quick" else [0.1, 1, 10]
    # Gamma is relevant only for RBF; use separate grids to avoid redundant linear fits.
    grid = [
        {"nmf__n_components": ranks, "classifier__kernel": ["linear"], "classifier__C": costs},
        {
            "nmf__n_components": ranks,
            "classifier__kernel": ["rbf"],
            "classifier__C": costs,
            "classifier__gamma": ["scale"],
        },
    ]
    direct = Pipeline(
        [
            ("probes", TopVarianceSelector(n_features)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(class_weight="balanced", random_state=seed, dual="auto", max_iter=10000),
            ),
        ],
        memory=memory,
    )
    searches = {
        "nmf_svm": GridSearchCV(
            nmf,
            grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
            return_train_score=True,
        ),
        "linear_svm": GridSearchCV(
            direct,
            {"classifier__C": costs},
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
            return_train_score=True,
        ),
        "dummy": GridSearchCV(
            DummyClassifier(strategy="most_frequent"),
            {},
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
        ),
    }
    return searches, n_features, n_folds


def fit_benchmark(dataset, output_dir, profile="quick", seed=42):
    validate_dataset(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    started = time.perf_counter()
    train = dataset.cohorts["train"]
    searches, n_features, n_folds = build_searches(
        profile, seed, output_dir / "cache", demo=dataset.mode == "demo"
    )
    captured_warnings = []
    cv_frames = []
    # One BLAS thread prevents oversubscription and makes small NMF runs much faster.
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        for name, search in searches.items():
            print(
                f"Fitting {name}: {n_folds} training folds, {n_features} selected probes",
                flush=True,
            )
            search.fit(train.X.to_numpy(), train.y)
            results = pd.DataFrame(search.cv_results_)
            results.insert(0, "model", name)
            cv_frames.append(results)
        captured_warnings = sorted(
            {str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)}
        )
    winner = max(searches, key=lambda name: searches[name].best_score_)
    print(
        f"Selected {winner} by training CV macro F1: {searches[winner].best_score_:.4f}", flush=True
    )
    evaluations = []
    predictions = {}
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        for model_name, search in searches.items():
            for split, cohort in dataset.cohorts.items():
                if split == "train":
                    continue
                pred = search.predict(cohort.X.to_numpy())
                record = evaluate(cohort.y, pred)
                record.update({"model": model_name, "split": split})
                evaluations.append(record)
                frame = pd.DataFrame({"actual": cohort.y, "predicted": pred}, index=cohort.X.index)
                frame["correct"] = frame.actual == frame.predicted
                frame.to_csv(
                    output_dir / "predictions" / f"{model_name}_{split}.csv",
                    index_label="sample_id",
                )
                predictions[model_name, split] = frame
        captured_warnings.extend(
            sorted({str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)})
        )
    hm = predictions[winner, "hm450_validation"]
    epic = predictions[winner, "epic_validation"]
    ids = hm.index.intersection(epic.index)
    paired = {
        "n_pairs": len(ids),
        "prediction_agreement": float(
            np.mean(hm.loc[ids, "predicted"] == epic.loc[ids, "predicted"])
        ),
    }
    selection = {
        "winner": winner,
        "cv_macro_f1": float(searches[winner].best_score_),
        "best_params": searches[winner].best_params_,
        "candidates": {
            name: {"cv_macro_f1": float(s.best_score_), "best_params": s.best_params_}
            for name, s in searches.items()
        },
        "cv_folds": n_folds,
        "max_selected_probes": n_features,
    }
    summary = {
        "project_version": __version__,
        "mode": dataset.mode,
        "profile": profile,
        "seed": seed,
        "data_summary": dataset.summary,
        "selection": selection,
        "evaluations": evaluations,
        "paired_validation": paired,
        "convergence_warnings": sorted(set(captured_warnings)),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    simple = [{k: v for k, v in r.items() if isinstance(v, (str, int, float))} for r in evaluations]
    pd.DataFrame(simple).to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(cv_frames, ignore_index=True).to_csv(output_dir / "cv_results.csv", index=False)
    bundle = {
        "pipeline": searches[winner].best_estimator_,
        "feature_names": train.X.columns.tolist(),
        "labels": LABELS,
        "selection": selection,
        "mode": dataset.mode,
        "project_version": __version__,
    }
    joblib.dump(bundle, output_dir / "model.joblib", compress=3)
    if winner != "dummy":
        selected = train.X.columns[
            searches[winner].best_estimator_.named_steps["probes"].get_support()
        ]
        pd.Series(selected, name="probe_id").to_csv(output_dir / "selected_probes.csv", index=False)
    manifests = [
        pd.DataFrame({"sample_id": c.X.index, "subtype": c.y.to_numpy(), "split": split})
        for split, c in dataset.cohorts.items()
    ]
    pd.concat(manifests, ignore_index=True).to_csv(output_dir / "split_manifest.csv", index=False)
    return summary
