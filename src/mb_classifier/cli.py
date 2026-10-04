"""Small, explicit commands for demo, audit, training, reporting, and prediction."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from mb_classifier.data import INPUT_FILES, load_prepared_data, make_demo_data, validate_beta_matrix
from mb_classifier.modeling import fit_benchmark


def input_manifest(data_dir):
    records = {}
    for name in INPUT_FILES:
        path = Path(data_dir) / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return records


def predict_file(model_path, input_path, output_path, id_column="sample_id"):
    # joblib is executable serialization: load only a bundle produced by a trusted run.
    bundle = joblib.load(model_path)
    with Path(input_path).open(newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Prediction CSV headers must be unique; duplicate probe names found")
    if id_column not in header:
        raise ValueError(f"Prediction CSV must contain the specimen ID column '{id_column}'")
    frame = pd.read_csv(input_path, dtype={id_column: str}).set_index(id_column)
    required = bundle["feature_names"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Input lacks {len(missing)} required probes, e.g. {missing[:3]}")
    frame = frame.loc[:, required]
    validate_beta_matrix(frame)
    with threadpool_limits(limits=1):
        pred = bundle["pipeline"].predict(frame.to_numpy())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predicted_subtype": pred}, index=frame.index).to_csv(
        output, index_label=id_column
    )
    return output


def _parser():
    parser = argparse.ArgumentParser(description="Medulloblastoma methylation benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "train"):
        item = sub.add_parser(command)
        item.add_argument("--output", type=Path, default=Path(f"reports/{command}"))
        item.add_argument("--seed", type=int, default=42)
        item.add_argument("--profile", choices=["quick", "full"], default="quick")
        if command == "train":
            item.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    audit = sub.add_parser("audit")
    audit.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    audit.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    report = sub.add_parser("report")
    report.add_argument("--output", type=Path, required=True, help="Existing run directory")
    pred = sub.add_parser("predict")
    pred.add_argument("--model", type=Path, required=True)
    pred.add_argument("--input", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--id-column", default="sample_id")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "predict":
            print(predict_file(args.model, args.input, args.output, args.id_column))
            return
        if args.command == "report":
            from mb_classifier.reporting import render_report

            render_report(args.output)
            print(args.output / "index.html")
            return
        dataset = (
            make_demo_data(args.seed)
            if args.command == "demo"
            else load_prepared_data(args.data_dir)
        )
        if args.command == "audit":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            content = {"data_summary": dataset.summary, "inputs": input_manifest(args.data_dir)}
            args.output.write_text(json.dumps(content, indent=2) + "\n")
            print(args.output)
            return
        summary = fit_benchmark(dataset, args.output, args.profile, args.seed)
        environment = {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "pyarrow",
                "joblib",
            ]
        }
        environment["python"] = sys.version
        (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        if args.command == "train":
            (args.output / "input_manifest.json").write_text(
                json.dumps(input_manifest(args.data_dir), indent=2) + "\n"
            )
        from mb_classifier.reporting import render_report

        render_report(args.output)
        print(f"Report: {args.output / 'index.html'}")
        for result in summary["evaluations"]:
            if result["model"] == summary["selection"]["winner"]:
                print(
                    f"{result['split']}: macro F1 {result['macro_f1']:.4f}, "
                    f"accuracy {result['accuracy']:.4f}"
                )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
