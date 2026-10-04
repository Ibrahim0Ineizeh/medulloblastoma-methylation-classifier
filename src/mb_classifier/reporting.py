"""Render a reproducible, offline results dashboard from aggregate metrics.

The report intentionally contains no sample identifiers or feature-level data.
Report figures use their own non-interactive canvas, so generation works on
headless machines without changing a notebook's configured plotting backend.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mb-classifier-matplotlib"))

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

NAVY = "#142b3e"
TEAL = "#087f8c"
GOLD = "#c88a2d"
SLATE = "#728696"
GRID = "#dce6e9"
SPLIT_NAMES = {
    "train": "Training set",
    "test": "Held-out test set",
    "hm450_validation": "HM450-compatible validation",
    "epic_validation": "EPIC validation",
}
MODEL_NAMES = {
    "nmf_svm": "NMF + SVM",
    "linear_svm": "Linear SVM",
    "dummy": "Dummy baseline",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent(value: Any) -> str:
    number = _number(value)
    return f"{number:.1%}" if number is not None else "—"


def _count(value: Any) -> str:
    number = _number(value)
    return f"{int(number):,}" if number is not None else "—"


def _model_name(value: Any) -> str:
    key = str(value)
    return MODEL_NAMES.get(key, key.replace("_", " "))


def _split_name(value: Any) -> str:
    key = str(value)
    return SPLIT_NAMES.get(key, key.replace("_", " ").capitalize())


def _class_name(value: Any) -> str:
    label = str(value)
    return f"Subtype {label}" if label.isdigit() else label


def _slug(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-") or "result"


def _interval(evaluation: dict[str, Any]) -> str:
    bounds = evaluation.get("accuracy_ci95")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return "—"
    return f"{_percent(bounds[0])}–{_percent(bounds[1])}"


def _plot_style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": NAVY,
        "axes.edgecolor": GRID,
        "axes.titlecolor": NAVY,
        "text.color": NAVY,
        "xtick.color": SLATE,
        "ytick.color": NAVY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def _figure(**kwargs: Any) -> Figure:
    """Create a headless report figure without registering it with pyplot."""
    figure = Figure(**kwargs)
    FigureCanvasAgg(figure)
    return figure


def _save_figure(figure: Figure, destination: Path) -> None:
    figure.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.22)
    figure.clear()


def _comparison_figure(evaluations: list[dict[str, Any]], winner: str, destination: Path) -> bool:
    comparisons = [row for row in evaluations if row.get("split") != "train"]
    if not comparisons:
        comparisons = evaluations
    splits = list(dict.fromkeys(str(row.get("split", "unknown")) for row in comparisons))
    if not splits:
        return False
    with matplotlib.rc_context(_plot_style()):
        figure = _figure(figsize=(10.4, 2.0 + 2.0 * len(splits)))
        axes = figure.subplots(len(splits), 1, squeeze=False)
        for split, axis in zip(splits, axes.flat):
            rows = [row for row in comparisons if str(row.get("split", "unknown")) == split]
            rows.sort(key=lambda row: _number(row.get("macro_f1")) or 0, reverse=True)
            positions = np.arange(len(rows))
            scores = [_number(row.get("macro_f1")) or 0 for row in rows]
            colors = [TEAL if row.get("model") == winner else SLATE for row in rows]
            axis.barh(positions, scores, color=colors, height=0.55)
            axis.set_yticks(positions, [_model_name(row.get("model", "Unknown")) for row in rows])
            axis.invert_yaxis()
            axis.set_xlim(0, 1.13)
            axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
            axis.set_title(_split_name(split), loc="left", fontsize=12, fontweight="bold", pad=12)
            axis.set_axisbelow(True)
            axis.grid(axis="x", color=GRID, linewidth=0.8)
            axis.tick_params(axis="both", length=0)
            axis.spines["left"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            for position, score, row in zip(positions, scores, rows):
                axis.text(
                    score + 0.02,
                    position,
                    _percent(row.get("macro_f1")),
                    va="center",
                    fontsize=11,
                    color=NAVY,
                )
        axes[-1, 0].set_xlabel("Macro F1 · each subtype contributes equally", labelpad=12)
        figure.tight_layout(h_pad=2.3)
        _save_figure(figure, destination)
    return True


def _confusion_figure(evaluation: dict[str, Any], destination: Path) -> bool:
    labels = [str(label) for label in evaluation.get("labels", [])]
    matrix = np.asarray(evaluation.get("confusion_matrix", []), dtype=float)
    if not labels or matrix.shape != (len(labels), len(labels)):
        return False
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    palette = LinearSegmentedColormap.from_list("mb_confusion", ["#f3f8f8", "#68b6bb", NAVY])
    with matplotlib.rc_context(_plot_style()):
        side = max(5.3, len(labels) * 0.57 + 1.5)
        figure = _figure(figsize=(side, side - 0.3))
        axis = figure.subplots()
        axis.imshow(normalized, cmap=palette, vmin=0, vmax=1)
        names = [_class_name(label) for label in labels]
        axis.set_xticks(range(len(labels)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), names)
        axis.set_xlabel("Predicted subtype", labelpad=12)
        axis.set_ylabel("True subtype", labelpad=10)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(labels)):
            for column in range(len(labels)):
                count = matrix[row, column]
                annotation = f"{int(count)}\n{normalized[row, column]:.0%}" if count else "0"
                axis.text(
                    column,
                    row,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=9 if len(labels) <= 8 else 8,
                    color="white" if normalized[row, column] >= 0.62 else NAVY,
                )
        figure.tight_layout()
        _save_figure(figure, destination)
    return True


def _distribution_figure(class_counts: dict[str, Any], destination: Path) -> bool:
    if not class_counts:
        return False
    labels = sorted(
        class_counts,
        key=lambda label: (
            not str(label).isdigit(),
            int(label) if str(label).isdigit() else str(label),
        ),
    )
    counts = [_number(class_counts[label]) or 0 for label in labels]
    with matplotlib.rc_context(_plot_style()):
        figure = _figure(figsize=(8.4, max(3.2, len(labels) * 0.47)))
        axis = figure.subplots()
        positions = np.arange(len(labels))
        axis.barh(positions, counts, height=0.60, color=TEAL)
        axis.set_yticks(positions, [_class_name(label) for label in labels])
        axis.invert_yaxis()
        maximum = max(counts, default=0)
        axis.set_xlim(0, max(1, maximum * 1.22))
        axis.set_xlabel("Training samples", labelpad=12)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.tick_params(length=0)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        for position, count in zip(positions, counts):
            axis.text(
                count + max(1, maximum * 0.02),
                position,
                f"{int(count):,}",
                va="center",
                fontsize=11,
            )
        figure.tight_layout()
        _save_figure(figure, destination)
    return True


def _overview_figure(metrics: dict[str, Any], destination: Path) -> None:
    """Produce a compact, clearly labeled README overview image."""
    evaluations = metrics.get("evaluations", [])
    selection = metrics.get("selection", {})
    winner = str(selection.get("winner", "unknown"))
    selected = [row for row in evaluations if row.get("model") == winner]
    primary = next((row for row in selected if row.get("split") == "test"), None)
    primary = primary or next((row for row in selected if row.get("split") != "train"), {})
    summary = metrics.get("data_summary", {})
    demo = metrics.get("mode") != "real"
    with matplotlib.rc_context(_plot_style()):
        figure = _figure(figsize=(13.3, 7.5), facecolor="#f5f6f2")
        figure.text(0.055, 0.93, "MB / METHYLATION", color=TEAL, fontsize=10, weight="bold")
        figure.text(
            0.055, 0.866, "Medulloblastoma subtype classification", fontsize=25, weight="bold"
        )
        figure.text(
            0.055,
            0.817,
            "A reproducible benchmark of NMF + SVM, Linear SVM, and a dummy baseline",
            color=SLATE,
            fontsize=12,
        )
        stats = [
            (
                "CV-SELECTED MODEL",
                _model_name(winner),
                f"CV macro F1: {_percent(selection.get('cv_macro_f1'))}",
            ),
            (
                "EVALUATION MACRO F1",
                _percent(primary.get("macro_f1")),
                _split_name(primary.get("split", "Evaluation")),
            ),
            (
                "PREPARED INPUT FEATURES",
                _count(summary.get("n_features")),
                f"{_count(summary.get('splits', {}).get('train', {}).get('n_samples'))} training samples",
            ),
        ]
        for position, (title, value, description) in zip([0.055, 0.365, 0.675], stats):
            card = FancyBboxPatch(
                (position, 0.65),
                0.27,
                0.12,
                boxstyle="round,pad=0.015",
                linewidth=0.8,
                edgecolor=GRID,
                facecolor="white",
                transform=figure.transFigure,
            )
            figure.add_artist(card)
            figure.text(position + 0.005, 0.739, title, fontsize=9, color=SLATE, weight="bold")
            figure.text(position + 0.005, 0.696, value, fontsize=22, color=NAVY, weight="bold")
            figure.text(position + 0.005, 0.663, description, fontsize=9, color=SLATE)

        axis = figure.add_axes([0.165, 0.22, 0.37, 0.32], facecolor="#f5f6f2")
        rows = [row for row in evaluations if row.get("split") == primary.get("split")]
        rows.sort(key=lambda row: _number(row.get("macro_f1")) or 0, reverse=True)
        positions = np.arange(len(rows))
        scores = [_number(row.get("macro_f1")) or 0 for row in rows]
        axis.barh(
            positions,
            scores,
            height=0.52,
            color=[TEAL if row.get("model") == winner else SLATE for row in rows],
        )
        axis.set_yticks(positions, [_model_name(row.get("model", "Unknown")) for row in rows])
        axis.invert_yaxis()
        axis.set_xlim(0, 1.13)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.50", "0.75", "1.00"])
        axis.set_xlabel("Macro F1", labelpad=10)
        axis.tick_params(length=0)
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        for position, score in zip(positions, scores):
            axis.text(
                score + 0.02, position, _percent(score), va="center", fontsize=10, weight="bold"
            )
        figure.text(
            0.055,
            0.575,
            _split_name(primary.get("split", "Evaluation")),
            fontsize=13,
            weight="bold",
        )

        confusion_axis = figure.add_axes([0.64, 0.20, 0.28, 0.34])
        matrix = np.asarray(primary.get("confusion_matrix", []), dtype=float)
        labels = [str(label) for label in primary.get("labels", [])]
        if labels and matrix.shape == (len(labels), len(labels)):
            totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
            palette = LinearSegmentedColormap.from_list("mb_overview", ["#f3f8f8", "#68b6bb", NAVY])
            confusion_axis.imshow(normalized, cmap=palette, vmin=0, vmax=1)
            confusion_axis.set_xticks(range(len(labels)), labels)
            confusion_axis.set_yticks(range(len(labels)), labels)
            confusion_axis.set_xlabel("Predicted subtype", fontsize=10)
            confusion_axis.set_ylabel("True subtype", fontsize=10)
            confusion_axis.tick_params(length=0, labelsize=8)
            for row in range(len(labels)):
                for column in range(len(labels)):
                    confusion_axis.text(
                        column,
                        row,
                        f"{int(matrix[row, column])}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if normalized[row, column] >= 0.62 else NAVY,
                    )
        else:
            confusion_axis.axis("off")
            confusion_axis.text(0.5, 0.5, "No confusion matrix available", ha="center", va="center")
        for spine in confusion_axis.spines.values():
            spine.set_visible(False)
        figure.text(0.64, 0.575, "Selected model · errors by subtype", fontsize=13, weight="bold")
        figure.text(
            0.64,
            0.12,
            "Cell values are sample counts; color shows true-row proportion.",
            color=SLATE,
            fontsize=8,
        )

        footer_color = "#fff0d2" if demo else "#e5f2f0"
        footer = FancyBboxPatch(
            (0.055, 0.035),
            0.89,
            0.047,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=footer_color,
            transform=figure.transFigure,
        )
        figure.add_artist(footer)
        footer_text = (
            "SYNTHETIC DEMONSTRATION  ·  Scores verify the workflow; they do not estimate performance on patient data."
            if demo
            else "PREPARED COHORT BENCHMARK  ·  Research use only; paired validation representations are not independent cohorts."
        )
        figure.text(0.066, 0.052, footer_text, fontsize=9, color=NAVY)
        _save_figure(figure, destination)


def _evaluation_rows(evaluations: list[dict[str, Any]], winner: str) -> str:
    rows = []
    for evaluation in evaluations:
        selected = evaluation.get("model") == winner
        model = _escape(_model_name(evaluation.get("model", "Unknown")))
        badge = '<span class="selected-label">CV selected</span>' if selected else ""
        rows.append(
            f'<tr class="{"selected-row" if selected else ""}">'
            f'<th scope="row">{model}{badge}</th>'
            f"<td>{_escape(_split_name(evaluation.get('split', 'Unknown')))}</td>"
            f'<td class="numeric">{_count(evaluation.get("n_samples"))}</td>'
            f'<td class="numeric">{_percent(evaluation.get("accuracy"))}</td>'
            f'<td class="numeric">{_percent(evaluation.get("balanced_accuracy"))}</td>'
            f'<td class="numeric strong">{_percent(evaluation.get("macro_f1"))}</td>'
            f'<td class="numeric">{_escape(_interval(evaluation))}</td></tr>'
        )
    return "".join(rows)


def _per_class_tables(evaluations: list[dict[str, Any]], winner: str) -> str:
    blocks = []
    for evaluation in evaluations:
        if evaluation.get("model") != winner or evaluation.get("split") == "train":
            continue
        per_class = evaluation.get("per_class", {})
        rows = []
        for label in evaluation.get("labels", []):
            values = per_class.get(str(label), per_class.get(label, {}))
            if not isinstance(values, dict):
                continue
            rows.append(
                f'<tr><th scope="row">{_escape(_class_name(label))}</th>'
                f'<td class="numeric">{_percent(values.get("precision"))}</td>'
                f'<td class="numeric">{_percent(values.get("recall"))}</td>'
                f'<td class="numeric">{_percent(values.get("f1-score", values.get("f1")))}</td>'
                f'<td class="numeric">{_count(values.get("support"))}</td></tr>'
            )
        if rows:
            blocks.append(
                f"<details><summary>{_escape(_split_name(evaluation.get('split')))} · subtype metrics</summary>"
                '<div class="table-scroll"><table><caption class="sr-only">Precision, recall, F1 and support by subtype</caption>'
                '<thead><tr><th scope="col">Subtype</th><th scope="col" class="numeric">Precision</th>'
                '<th scope="col" class="numeric">Recall</th><th scope="col" class="numeric">F1</th>'
                '<th scope="col" class="numeric">Support</th></tr></thead>'
                f"<tbody>{''.join(rows)}</tbody></table></div></details>"
            )
    return "".join(blocks)


CSS = """
:root { --navy:#142b3e; --teal:#087f8c; --ink:#253f50; --muted:#627783; --paper:#f5f6f2; --line:#dce6e5; --gold:#c88a2d; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--paper); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; font-size:16px; line-height:1.6; }
a { color:var(--teal); text-underline-offset:3px; }
a:hover { color:var(--navy); }
a:focus-visible, summary:focus-visible { outline:3px solid var(--gold); outline-offset:5px; }
.skip { position:absolute; top:-80px; left:16px; padding:12px 20px; background:white; z-index:5; }
.skip:focus { top:12px; }
.wrap { max-width:1220px; margin:0 auto; padding:0 36px; }
.topbar { border-bottom:1px solid var(--line); background:#fff; }
.topbar .wrap { display:flex; justify-content:space-between; align-items:center; gap:24px; min-height:80px; }
.brand { display:flex; align-items:center; gap:13px; color:var(--navy); font-weight:750; letter-spacing:-.02em; font-size:18px; }
.brand-mark { width:34px; height:34px; border-radius:9px; background:var(--navy); display:grid; grid-template-columns:repeat(3,4px); gap:4px; justify-content:center; align-content:center; }
.brand-mark i { width:4px; height:14px; background:#7fc7c8; border-radius:2px; }
.brand-mark i:nth-child(2) { height:20px; background:white; transform:translateY(-3px); }
nav { display:flex; gap:24px; font-size:13px; font-weight:650; }
nav a { text-decoration:none; color:var(--muted); }
.hero { padding:70px 0 40px; }
.eyebrow { font-size:12px; text-transform:uppercase; letter-spacing:.16em; font-weight:750; color:var(--teal); margin:0 0 18px; }
.hero-grid { display:grid; grid-template-columns:1fr 270px; gap:64px; align-items:start; }
h1 { margin:0 0 22px; font-family:Georgia,"Times New Roman",serif; font-weight:400; font-size:clamp(40px,5.2vw,67px); letter-spacing:-.045em; line-height:1.04; color:var(--navy); }
.lede { font-size:18px; max-width:680px; margin:0; color:var(--muted); line-height:1.7; }
.study-card { background:var(--navy); color:white; border-radius:16px; padding:25px 26px; margin-top:8px; }
.study-card .eyebrow { color:#93cfd1; margin-bottom:17px; }
.study-card dl { margin:0; }
.study-card dl > div { display:flex; justify-content:space-between; gap:15px; border-top:1px solid #385064; padding:10px 0; font-size:13px; }
.study-card dt { color:#bed0dc; }
.study-card dd { margin:0; font-weight:650; text-align:right; }
.badge { display:inline-flex; border:1px solid #9bbcc0; color:var(--teal); background:#edf6f5; padding:5px 12px; border-radius:100px; font-weight:700; font-size:12px; }
.notice { display:flex; align-items:flex-start; gap:16px; padding:22px 26px; border:1px solid #e2cdab; border-left:4px solid var(--gold); border-radius:10px; background:#fff9ed; font-size:14px; }
.notice p { margin:0; }
.notice strong { color:var(--navy); }
.notice.cohort { background:#edf6f5; border-color:#b8d6d7; border-left-color:var(--teal); }
.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:17px; margin:26px 0 46px; }
.stat { background:white; border:1px solid var(--line); border-radius:12px; padding:23px; }
.stat-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; font-weight:650; }
.stat-value { font-size:34px; line-height:1.18; letter-spacing:-.045em; color:var(--navy); margin:13px 0 9px; font-weight:750; }
.stat-value.model { font-size:25px; letter-spacing:-.03em; }
.stat-detail { font-size:12px; line-height:1.5; color:var(--muted); margin:0; }
section { margin:0 0 28px; }
.panel { background:white; border:1px solid var(--line); border-radius:16px; padding:32px; }
.section-head { display:flex; justify-content:space-between; align-items:start; gap:24px; margin-bottom:25px; }
.section-number { font-size:12px; color:var(--teal); font-weight:750; letter-spacing:.1em; margin:0 0 7px; }
h2 { font-size:25px; letter-spacing:-.035em; color:var(--navy); margin:0 0 8px; line-height:1.3; }
h3 { font-size:17px; letter-spacing:-.02em; color:var(--navy); margin:0 0 9px; }
p { margin:0 0 16px; }
.subtext { color:var(--muted); font-size:14px; margin:0; max-width:760px; }
.legend { white-space:nowrap; color:var(--muted); font-size:12px; padding-top:9px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--teal); margin-right:7px; }
figure { margin:0; }
figure img { display:block; width:100%; height:auto; }
figcaption { font-size:12px; color:var(--muted); padding-top:12px; line-height:1.65; }
.comparison-figure { max-width:1020px; margin:0 auto; }
.table-scroll { width:100%; overflow-x:auto; margin-top:26px; }
table { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; text-align:left; }
thead { color:var(--muted); background:#f3f7f7; }
th,td { padding:13px 14px; border-bottom:1px solid var(--line); vertical-align:middle; }
thead th { font-size:11px; text-transform:uppercase; letter-spacing:.04em; font-weight:700; white-space:nowrap; }
tbody th { font-weight:650; color:var(--navy); min-width:140px; }
.numeric { text-align:right; white-space:nowrap; }
.strong { font-weight:750; color:var(--navy); }
.selected-row { background:#f1f8f7; }
.selected-label { display:block; font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:var(--teal); margin-top:2px; }
.footnote { color:var(--muted); font-size:12px; margin:17px 0 0; }
.matrix-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); gap:24px; }
.matrix-card { border:1px solid var(--line); border-radius:12px; padding:21px 15px 16px; overflow:hidden; }
.matrix-card h3 { margin-left:7px; font-size:15px; }
.matrix-meta { margin:0 7px 12px; font-size:12px; color:var(--muted); }
.matrix-card figcaption { margin:0 7px; }
.details-group { margin-top:25px; }
details { border-top:1px solid var(--line); padding:16px 0; }
summary { cursor:pointer; font-weight:650; color:var(--navy); font-size:14px; }
details .table-scroll { margin-top:16px; }
.data-grid { display:grid; grid-template-columns:1.05fr 1fr; gap:38px; }
.data-grid .table-scroll { margin-top:0; }
.data-grid th,.data-grid td { padding:12px 8px; }
.class-counts { font-size:11px; line-height:1.65; color:var(--muted); display:block; margin-top:5px; font-weight:400; }
.pair-note { background:#f1f7f7; border-radius:10px; padding:19px 23px; margin-top:24px; font-size:14px; border-left:3px solid var(--teal); }
.pair-note p:last-child { margin-bottom:0; }
.method-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:28px; margin-top:24px; }
.method-step { border-left:2px solid var(--line); padding-left:20px; }
.method-step .step { color:var(--teal); font-size:11px; letter-spacing:.1em; font-weight:750; display:block; margin-bottom:11px; }
.method-step p { font-size:13px; color:var(--muted); margin:0; }
pre { background:#f3f7f7; border:1px solid var(--line); border-radius:9px; padding:20px; overflow-x:auto; font-size:12px; color:var(--navy); line-height:1.7; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
.notes { font-size:13px; color:var(--muted); padding-left:20px; }
.notes li { margin:8px 0; }
.warning { margin-top:20px; font-size:13px; background:#fff9ed; padding:18px 22px; border-radius:9px; }
.warning ul { padding-left:20px; margin-bottom:0; }
.report-footer { padding:16px 0 40px; color:var(--muted); display:flex; justify-content:space-between; gap:25px; font-size:12px; }
.report-footer p { margin:0; max-width:690px; }
.sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
@media(max-width:1000px) { .hero-grid { gap:30px; grid-template-columns:1fr 250px; } .cards { grid-template-columns:repeat(2,1fr); } .matrix-grid { grid-template-columns:repeat(2,1fr); } .data-grid { grid-template-columns:1fr; gap:28px; } }
@media(max-width:680px) { .wrap { padding:0 20px; } .topbar .wrap { min-height:70px; } .brand { font-size:15px; } nav { gap:14px; font-size:11px; } nav a:last-child { display:none; } .hero { padding-top:42px; } .hero-grid { grid-template-columns:1fr; gap:25px; } h1 { font-size:45px; } .lede { font-size:16px; } .study-card { margin-top:0; } .study-card dl { display:grid; grid-template-columns:1fr 1fr; gap:0 20px; } .cards { gap:12px; margin-bottom:28px; } .stat { padding:19px 15px; } .stat-value { font-size:30px; } .stat-value.model { font-size:22px; } .panel { padding:23px 17px; } .section-head { display:block; } .legend { margin-top:14px; } h2 { font-size:23px; } .matrix-grid { grid-template-columns:1fr; } .method-grid { grid-template-columns:1fr; gap:23px; } .notice { padding:20px; } .report-footer { flex-direction:column; gap:14px; } th,td { padding:12px 10px; } }
@media(prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
@media print { body { background:white; font-size:10pt; } .wrap { max-width:none; padding:0; } .topbar,nav,.skip { display:none; } .hero { padding-top:15px; } h1 { font-size:36pt; } .panel { break-inside:avoid; } .cards { gap:10px; } a { color:var(--navy); } details { display:block; } details > * { display:block; } }
"""


def render_report(output_dir: Path) -> Path:
    """Write local figures and ``index.html`` using ``metrics.json``.

    Returns the HTML path. The report is portable with its ``figures`` directory
    and metrics file; it loads no scripts, stylesheets, fonts or other network
    assets. Missing optional validation results are omitted gracefully.
    """
    output_dir = Path(output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _overview_figure(metrics, figures_dir / "benchmark-overview.png")
    evaluations = metrics.get("evaluations", [])
    selection = metrics.get("selection", {})
    winner = str(selection.get("winner", "unknown"))
    winner_name = _model_name(winner)
    summary = metrics.get("data_summary", {})
    splits = summary.get("splits", {})
    mode = metrics.get("mode", "demo")
    demo = mode != "real"
    mode_name = "Synthetic demonstration" if demo else "Prepared cohort benchmark"
    profile = str(metrics.get("profile", "unknown")).capitalize()
    runtime = _number(metrics.get("runtime_seconds"))
    runtime_text = f"{runtime:.1f} s" if runtime is not None else "—"
    winner_results = [evaluation for evaluation in evaluations if evaluation.get("model") == winner]
    held_out = next(
        (evaluation for evaluation in winner_results if evaluation.get("split") == "test"), None
    )
    primary_result = held_out or next(
        (evaluation for evaluation in winner_results if evaluation.get("split") != "train"), {}
    )
    score_context = (
        "Held-out test" if held_out else _split_name(primary_result.get("split", "Evaluation"))
    )

    if demo:
        notice = (
            '<div class="notice" role="note"><p><strong>These are synthetic demonstration results.</strong> '
            "The samples are simulated to exercise the complete workflow. Scores here do not estimate "
            "performance on patient data and must not be interpreted as biological or clinical evidence.</p></div>"
        )
    else:
        notice = (
            '<div class="notice cohort" role="note"><p><strong>Research benchmark on prepared cohort inputs.</strong> '
            "These results describe classification using supplied representations and subtype labels. "
            "Upstream preparation limits how strongly the scores can support independent generalization; "
            "see the provenance and validation notes below.</p></div>"
        )

    comparison_exists = _comparison_figure(
        evaluations, winner, figures_dir / "model-comparison.png"
    )
    comparison = (
        '<figure class="comparison-figure"><img src="figures/model-comparison.png" '
        'alt="Macro F1 scores for each model on each evaluation split. Exact values appear in the table below." '
        'width="1200" loading="eager"><figcaption>Macro F1 is the unweighted mean of subtype F1 scores. '
        "The teal bars identify the model selected using training cross-validation.</figcaption></figure>"
        if comparison_exists
        else '<p class="subtext">No evaluation results are available.</p>'
    )
    matrix_cards = []
    matrix_results = [
        evaluation for evaluation in winner_results if evaluation.get("split") != "train"
    ]
    for index, evaluation in enumerate(matrix_results):
        filename = f"confusion-{_slug(evaluation.get('split', 'unknown'))}-{index}.png"
        if not _confusion_figure(evaluation, figures_dir / filename):
            continue
        split_name = _split_name(evaluation.get("split", "Unknown"))
        matrix_cards.append(
            '<figure class="matrix-card">'
            f"<h3>{_escape(split_name)}</h3>"
            f'<p class="matrix-meta">n = {_count(evaluation.get("n_samples"))} · Macro F1 {_percent(evaluation.get("macro_f1"))}</p>'
            f'<img src="figures/{_escape(filename)}" alt="Confusion matrix for {_escape(winner_name)} on '
            f'{_escape(split_name)}. Rows are true subtypes and columns are predicted subtypes." loading="lazy">'
            "<figcaption>Each cell shows the sample count and, when nonzero, the percentage of its true-subtype row.</figcaption></figure>"
        )
    matrices = (
        "".join(matrix_cards)
        or '<p class="subtext">No confusion matrices are available for the selected model.</p>'
    )

    distribution_exists = _distribution_figure(
        splits.get("train", {}).get("class_counts", {}), figures_dir / "training-distribution.png"
    )
    distribution = (
        '<figure><img src="figures/training-distribution.png" alt="Training sample counts for each subtype." '
        'loading="lazy"><figcaption>Training class balance provides context for the macro F1 and '
        "balanced accuracy metrics.</figcaption></figure>"
        if distribution_exists
        else '<p class="subtext">Training class counts are unavailable.</p>'
    )
    split_rows = []
    for split, values in splits.items():
        counts = values.get("class_counts", {})
        count_text = " · ".join(
            f"{_class_name(label)}: {_count(count)}" for label, count in counts.items()
        )
        split_rows.append(
            f'<tr><th scope="row">{_escape(_split_name(split))}'
            f'<span class="class-counts">{_escape(count_text)}</span></th>'
            f'<td class="numeric">{_count(values.get("n_samples"))}</td></tr>'
        )

    paired = metrics.get("paired_validation")
    if paired:
        paired_note = (
            '<div class="pair-note"><h3>Paired validation representations</h3>'
            f"<p>The two validation representations contain {_count(paired.get('n_pairs'))} matched tumors. "
            f"The selected model gives the same prediction for <strong>{_percent(paired.get('prediction_agreement'))}</strong> "
            "of matched pairs. Agreement measures prediction consistency; it does not establish correctness.</p>"
            "<p>HM450-compatible and EPIC validation scores must not be combined as independent evidence: "
            "they represent the same tumors in prepared platform representations.</p></div>"
        )
    else:
        paired_note = (
            '<div class="pair-note"><h3>Validation interpretation</h3><p>No matched-pair agreement was recorded. '
            "Validation splits should be interpreted using their provenance and sample relationships; "
            "multiple representations of the same tumor are not independent cohorts.</p></div>"
        )

    notes = summary.get("provenance_notes", [])
    notes_html = (
        '<ul class="notes">' + "".join(f"<li>{_escape(note)}</li>" for note in notes) + "</ul>"
        if notes
        else ""
    )
    warnings = metrics.get("convergence_warnings", [])
    warnings_html = (
        '<div class="warning"><strong>Optimization diagnostics</strong><ul>'
        + "".join(f"<li>{_escape(warning)}</li>" for warning in warnings)
        + '</ul><p class="footnote">Review these diagnostics before interpreting the corresponding fitted model.</p></div>'
        if warnings
        else ""
    )
    parameters = _escape(json.dumps(selection.get("best_params", {}), indent=2, sort_keys=True))
    evaluation_table = _evaluation_rows(evaluations, winner)
    per_class = _per_class_tables(evaluations, winner)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A reproducible research benchmark for Group 3/4 medulloblastoma subtype classification from prepared DNA methylation features.">
<title>Medulloblastoma methylation classifier · Results report</title>
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#main">Skip to results</a>
<header class="topbar"><div class="wrap">
<div class="brand"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>MB / methylation</div>
<nav aria-label="Report navigation"><a href="#results">Results</a><a href="#data">Data</a><a href="#method">Method</a><a href="metrics.json">Metrics JSON</a></nav>
</div></header>
<main id="main" class="wrap">
<div class="hero"><div class="hero-grid"><div>
<p class="eyebrow">Computational oncology / Research portfolio</p>
<h1>Medulloblastoma<br>methylation classification.</h1>
<p class="lede">A reproducible comparison of dimensionality reduction, support vector machines, and a baseline for Group 3/4 molecular subtype prediction.</p>
</div><aside class="study-card" aria-label="Run metadata">
<p class="eyebrow">The benchmark run</p><dl>
<div><dt>Dataset</dt><dd>{"Synthetic" if demo else "Prepared cohort"}</dd></div>
<div><dt>Profile</dt><dd>{_escape(profile)}</dd></div>
<div><dt>Random seed</dt><dd>{_escape(metrics.get("seed", "—"))}</dd></div>
<div><dt>Runtime</dt><dd>{_escape(runtime_text)}</dd></div>
</dl></aside></div></div>
{notice}
<div class="cards" aria-label="Key results">
<div class="stat"><div class="stat-label">CV-selected model</div><div class="stat-value model">{_escape(winner_name)}</div><p class="stat-detail">Training CV macro F1: {_percent(selection.get("cv_macro_f1"))}</p></div>
<div class="stat"><div class="stat-label">Evaluation macro F1</div><div class="stat-value">{_percent(primary_result.get("macro_f1"))}</div><p class="stat-detail">{_escape(score_context)} · equal weight per subtype</p></div>
<div class="stat"><div class="stat-label">Evaluation accuracy</div><div class="stat-value">{_percent(primary_result.get("accuracy"))}</div><p class="stat-detail">95% CI: {_escape(_interval(primary_result))}</p></div>
<div class="stat"><div class="stat-label">Input features</div><div class="stat-value">{_count(summary.get("n_features"))}</div><p class="stat-detail">{_count(splits.get("train", {}).get("n_samples"))} training samples · {_escape(mode_name)}</p></div>
</div>
<section id="results" class="panel" aria-labelledby="results-title">
<div class="section-head"><div><p class="section-number">01 / Model comparison</p><h2 id="results-title">Performance across evaluation splits</h2>
<p class="subtext">Model selection uses training cross-validation. Held-out and validation scores assess the selected model after tuning and are reported alongside the alternatives.</p></div>
<div class="legend"><span class="dot" aria-hidden="true"></span>Selected by training CV</div></div>
{comparison}
<div class="table-scroll"><table><caption class="sr-only">Aggregate model performance on each split</caption>
<thead><tr><th scope="col">Model</th><th scope="col">Evaluation split</th><th scope="col" class="numeric">Samples</th><th scope="col" class="numeric">Accuracy</th><th scope="col" class="numeric">Balanced acc.</th><th scope="col" class="numeric">Macro F1</th><th scope="col" class="numeric">Accuracy 95% CI</th></tr></thead>
<tbody>{evaluation_table}</tbody></table></div>
<p class="footnote">Accuracy measures total correct predictions. Balanced accuracy averages subtype recall; macro F1 balances precision and recall with equal subtype weighting. Confidence intervals describe sampling uncertainty in accuracy, not the effect of upstream preprocessing or dataset shift.</p>
</section>
<section class="panel" aria-labelledby="confusion-title">
<div class="section-head"><div><p class="section-number">02 / Selected model</p><h2 id="confusion-title">Where the classifier succeeds and struggles</h2>
<p class="subtext">Confusion matrices for {_escape(winner_name)}. Darker diagonal cells indicate correct classification within a subtype; off-diagonal counts show errors.</p></div></div>
<div class="matrix-grid">{matrices}</div>
<div class="details-group">{per_class}</div>
</section>
<section id="data" class="panel" aria-labelledby="data-title">
<div class="section-head"><div><p class="section-number">03 / Data context</p><h2 id="data-title">The sample structure behind the scores</h2>
<p class="subtext">Subtype labels describe molecular subtypes within Group 3/4 medulloblastoma. Counts and validation relationships matter when interpreting performance.</p></div></div>
<div class="data-grid">{distribution}<div><div class="table-scroll"><table><caption class="sr-only">Sample counts by split and subtype</caption><thead><tr><th scope="col">Split and subtype counts</th><th scope="col" class="numeric">Samples</th></tr></thead><tbody>{"".join(split_rows)}</tbody></table></div></div></div>
{paired_note}
{notes_html}
</section>
<section id="method" class="panel" aria-labelledby="method-title">
<p class="section-number">04 / Reproducibility &amp; interpretation</p><h2 id="method-title">A transparent experimental workflow</h2>
<p class="subtext">The report records aggregate results from a seeded run. It is generated directly from <a href="metrics.json">metrics.json</a> and uses local figures, so it can be read offline.</p>
<div class="method-grid">
<div class="method-step"><span class="step">STEP 01</span><h3>Inspect prepared inputs</h3><p>Check the feature matrix, subtype labels, sample counts, and split relationships before fitting classifiers.</p></div>
<div class="method-step"><span class="step">STEP 02</span><h3>Tune on training data</h3><p>Compare the baseline and SVM pipelines using cross-validation. Select the model by macro F1, with transformations fitted within the modeling pipeline.</p></div>
<div class="method-step"><span class="step">STEP 03</span><h3>Evaluate and document</h3><p>Report held-out and validation metrics, subtype errors, accuracy uncertainty, and paired-representation agreement where available.</p></div>
</div>
<details style="margin-top:28px"><summary>Selected model parameters</summary><pre><code>{parameters}</code></pre></details>
{warnings_html}
<p class="footnote"><strong>Research use only.</strong> This project is an educational research benchmark. It is not a validated diagnostic tool, and its predictions must not guide clinical decisions. Independent validation with a fully specified preprocessing workflow would be required for stronger generalization claims.</p>
</section>
<footer class="report-footer"><p>Medulloblastoma methylation classifier · {_escape(mode_name)} · {_escape(profile)} profile · Seed {_escape(metrics.get("seed", "—"))}</p><p><a href="metrics.json">Inspect the aggregate run record →</a></p></footer>
</main>
</body>
</html>
"""
    destination = output_dir / "index.html"
    destination.write_text(document, encoding="utf-8")
    return destination
