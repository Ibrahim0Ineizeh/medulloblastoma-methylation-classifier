import json
import subprocess
import sys

import joblib
import numpy as np
import pytest

from mb_classifier.cli import main, predict_file
from mb_classifier.data import make_demo_data
from mb_classifier.modeling import accuracy_interval, fit_benchmark
from mb_classifier.reporting import render_report


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    output = tmp_path_factory.mktemp("demo")
    dataset = make_demo_data(42)
    summary = fit_benchmark(dataset, output, seed=42)
    render_report(output)
    return output, dataset, summary


def test_complete_demo_artifacts_and_training_selection(run):
    output, dataset, summary = run
    assert (output / "index.html").is_file()
    assert len(list((output / "figures").glob("*.png"))) >= 3
    assert len(summary["evaluations"]) == 9
    assert summary["paired_validation"]["n_pairs"] == 24
    candidates = summary["selection"]["candidates"]
    assert summary["selection"]["cv_macro_f1"] == max(x["cv_macro_f1"] for x in candidates.values())
    assert set(dataset.cohorts["train"].X.index).isdisjoint(dataset.cohorts["test"].X.index)
    assert json.loads((output / "metrics.json").read_text())["mode"] == "demo"


@pytest.mark.parametrize("backend", ["Agg", "module://matplotlib_inline.backend_inline"])
def test_reporting_preserves_active_backend_and_figures(run, tmp_path, backend):
    _, _, summary = run
    (tmp_path / "metrics.json").write_text(json.dumps(summary))
    # A fresh process exercises the first import, as in a notebook kernel.
    code = """
import sys
import matplotlib
matplotlib.use(sys.argv[2])
import matplotlib.pyplot as plt
figure, axis = plt.subplots()
axis.plot([0, 1], [0, 1])
backend = matplotlib.get_backend()
canvas = figure.canvas
figures = plt.get_fignums()
from mb_classifier.reporting import render_report
assert matplotlib.get_backend() == backend, "Reporting import changed the backend"
render_report(sys.argv[1])
assert matplotlib.get_backend() == backend, "Reporting render changed the backend"
assert plt.get_fignums() == figures, "Reporting changed existing pyplot figures"
assert figure.canvas is canvas, "Reporting replaced an existing figure's canvas"
plt.close(figure)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), backend],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "index.html").is_file()
    figures = list((tmp_path / "figures").glob("*.png"))
    assert len(figures) == 6
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures)


def test_saved_model_predictions_match_and_probe_order_is_restored(run, tmp_path):
    output, dataset, summary = run
    frame = dataset.cohorts["test"].X
    path = tmp_path / "profiles.csv"
    frame[frame.columns[::-1]].to_csv(path, index_label="sample_id")
    result = tmp_path / "predictions.csv"
    predict_file(output / "model.joblib", path, result)
    import pandas as pd

    actual = pd.read_csv(result, dtype=str).predicted_subtype.to_numpy()
    bundle = joblib.load(output / "model.joblib")
    expected = bundle["pipeline"].predict(frame.to_numpy())
    assert np.array_equal(actual, expected)


def test_missing_prediction_features_fail(run, tmp_path):
    output, dataset, _ = run
    path = tmp_path / "incomplete.csv"
    dataset.cohorts["test"].X.iloc[:, :-1].to_csv(path, index_label="sample_id")
    with pytest.raises(ValueError, match="required probes"):
        predict_file(output / "model.joblib", path, tmp_path / "out.csv")


def test_duplicate_prediction_headers_fail_before_pandas_mangles_them(run, tmp_path):
    output, dataset, _ = run
    path = tmp_path / "duplicate.csv"
    frame = dataset.cohorts["test"].X.copy()
    frame.columns = [frame.columns[0], frame.columns[0], *frame.columns[2:]]
    frame.to_csv(path, index_label="sample_id")
    with pytest.raises(ValueError, match="headers must be unique"):
        predict_file(output / "model.joblib", path, tmp_path / "out.csv")


def test_blank_prediction_id_fails(run, tmp_path):
    output, dataset, _ = run
    path = tmp_path / "blank_id.csv"
    frame = dataset.cohorts["test"].X.copy()
    frame.index = ["", *frame.index[1:]]
    frame.to_csv(path, index_label="sample_id")
    with pytest.raises(ValueError, match="nonmissing and nonblank"):
        predict_file(output / "model.joblib", path, tmp_path / "out.csv")


def test_accuracy_interval_and_missing_data_cli(tmp_path):
    low, high = accuracy_interval(95, 100)
    assert 0 < low < 0.95 < high < 1
    with pytest.raises(SystemExit, match="Missing prepared inputs"):
        main(["train", "--data-dir", str(tmp_path)])
