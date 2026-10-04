"""Validate prepared methylation inputs and preserve specimen-level splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [str(i) for i in range(1, 9)]
INPUT_FILES = (
    "hm450_final.parquet",
    "epic_final.parquet",
    "gse130051_metadata.csv",
    "Train_Set1_MB_CommonProbes.csv",
)


@dataclass
class Cohort:
    X: pd.DataFrame
    y: pd.Series


@dataclass
class Dataset:
    cohorts: dict[str, Cohort]
    summary: dict
    mode: str


def validate_beta_matrix(frame):
    """Reject silent alignment errors, missing values, and invalid beta values."""
    if frame.empty or not len(frame.columns):
        raise ValueError("Beta matrix is empty")
    if frame.index.isna().any() or (frame.index.astype(str).str.strip() == "").any():
        raise ValueError("Specimen IDs must be nonmissing and nonblank")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Specimen IDs and probe names must be unique")
    if not all(str(c).startswith("cg") for c in frame.columns):
        raise ValueError("Every feature must be a CpG probe; remove target/metadata columns")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Beta values must be finite with no missing values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Methylation beta values must lie within [0, 1]")


def _summary(cohorts, notes):
    return {
        "n_features": int(cohorts["train"].X.shape[1]),
        "splits": {
            name: {"n_samples": len(c.y), "class_counts": c.y.value_counts().sort_index().to_dict()}
            for name, c in cohorts.items()
        },
        "provenance_notes": notes,
    }


def validate_dataset(dataset):
    train = dataset.cohorts["train"]
    if set(train.y) != set(LABELS):
        raise ValueError("Training data must contain all eight subtypes (1-8)")
    names = train.X.columns
    for name, c in dataset.cohorts.items():
        validate_beta_matrix(c.X)
        if not c.X.index.equals(c.y.index):
            raise ValueError(f"Labels and specimens are misaligned in {name}")
        if not c.X.columns.equals(names):
            raise ValueError(f"Probe order differs in {name}")
        if c.y.isna().any() or not set(c.y).issubset(LABELS):
            raise ValueError(f"Invalid subtype labels in {name}")
    holdouts = ["test", "hm450_validation", "epic_validation"]
    for name in holdouts:
        if len(train.X.index.intersection(dataset.cohorts[name].X.index)):
            raise ValueError(f"Training specimens overlap with {name}")
    test_ids = dataset.cohorts["test"].X.index
    for name in holdouts[1:]:
        if len(test_ids.intersection(dataset.cohorts[name].X.index)):
            raise ValueError(f"Test specimens overlap with {name}")
    hm, epic = [dataset.cohorts[n] for n in holdouts[1:]]
    if not epic.X.index.isin(hm.X.index).all():
        raise ValueError("EPIC validation must be a subset of the paired validation IDs")
    if not hm.y.loc[epic.y.index].equals(epic.y):
        raise ValueError("Paired validation subtype labels disagree")


def load_prepared_data(data_dir):
    data_dir = Path(data_dir)
    missing = [name for name in INPUT_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared inputs in {data_dir}: {', '.join(missing)}. "
            "See docs/DATA.md, or run 'mb-classifier demo' without downloads."
        )
    metadata = pd.read_csv(data_dir / INPUT_FILES[2], dtype={"gsm": str, "subtype": str})
    required = {"gsm", "platform", "subtype"}
    if not required.issubset(metadata.columns) or metadata.gsm.isna().any():
        raise ValueError("Metadata must contain nonmissing gsm, platform, and subtype")
    if metadata.gsm.duplicated().any():
        raise ValueError("Metadata specimen IDs must be unique")
    meta = metadata.set_index("gsm")
    hm = pd.read_parquet(data_dir / INPUT_FILES[0])
    epic = pd.read_parquet(data_dir / INPUT_FILES[1])
    hm.index, epic.index = hm.index.astype(str), epic.index.astype(str)
    for name, frame in [("HM450-compatible", hm), ("EPIC", epic)]:
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"Duplicate specimens or columns in {name}")
        if not frame.index.isin(meta.index).all():
            raise ValueError(f"{name} includes IDs absent from metadata")
        if "subtype" in frame:
            numeric_labels = pd.to_numeric(frame["subtype"], errors="raise")
            if not numeric_labels.isin(range(1, 9)).all():
                raise ValueError(f"Invalid embedded subtype labels in {name}")
            embedded = numeric_labels.astype(int).astype(str)
            if not embedded.equals(meta.loc[frame.index, "subtype"]):
                raise ValueError(f"Embedded subtype disagrees with metadata in {name}")
    # Read IDs and Group only. The inherited supervised probe panel is not reused.
    panel = pd.read_csv(data_dir / INPUT_FILES[3], usecols=["Samples", "Group"], dtype=str)
    train_ids = pd.Index(panel.Samples, name=hm.index.name)
    if train_ids.has_duplicates or not train_ids.isin(hm.index).all():
        raise ValueError("Training IDs must be unique and present in the prepared matrix")
    if not np.array_equal(panel.Group.to_numpy(), meta.loc[train_ids, "subtype"].to_numpy()):
        raise ValueError("Training manifest Group labels disagree with metadata")
    epic_metadata_ids = meta.index[(meta.platform == "GPL21145") & meta.subtype.isin(LABELS)]
    if len(train_ids.intersection(epic_metadata_ids)):
        raise ValueError("Training manifest includes EPIC validation specimens")
    if not (meta.loc[train_ids, "platform"] == "GPL13534").all():
        raise ValueError("Training specimens must belong to the HM450 metadata platform")
    if not (meta.loc[epic.index, "platform"] == "GPL21145").all():
        raise ValueError("EPIC matrix specimens must have EPIC platform metadata")
    input_probes = {
        "hm450": {column for column in hm.columns if str(column).startswith("cg")},
        "epic": {column for column in epic.columns if str(column).startswith("cg")},
    }
    probes = pd.Index(sorted(input_probes["hm450"] & input_probes["epic"]))
    test_ids = hm.index[
        (meta.loc[hm.index, "platform"].to_numpy() == "GPL13534") & ~hm.index.isin(train_ids)
    ].sort_values()
    validation_ids = hm.index[hm.index.isin(epic_metadata_ids)].sort_values()
    cohorts = {}
    for name, frame, ids in [
        ("train", hm, train_ids),
        ("test", hm, test_ids),
        ("hm450_validation", hm, validation_ids),
        ("epic_validation", epic, epic.index.sort_values()),
    ]:
        X = frame.loc[ids, probes].astype(np.float64)
        y = meta.loc[ids, "subtype"].rename("subtype")
        cohorts[name] = Cohort(X, y)
    notes = [
        "Prepared cohort benchmark; the upstream 22,892-probe curation is inherited.",
        "Only specimen IDs are reused from the legacy training panel; its probes are discarded.",
        "Validation sets are paired representations of overlapping tumors, "
        "not independent cohorts.",
        "All 99 labeled EPIC metadata IDs are reserved, including two without EPIC measurements.",
        "These historical assessment splits support retrospective evaluation, "
        "not a new untouched test.",
    ]
    summary = _summary(cohorts, notes)
    summary["probe_alignment"] = {
        "common_cpgs": len(probes),
        "representations": {
            name: {"input_cpgs": len(columns), "excluded_cpgs": len(columns) - len(probes)}
            for name, columns in input_probes.items()
        },
    }
    dataset = Dataset(cohorts, summary, "real")
    validate_dataset(dataset)
    return dataset


def make_demo_data(seed=42):
    """Create artificial beta profiles for a portable engineering demonstration."""
    rng = np.random.default_rng(seed)
    n_features = 96
    columns = [f"cg{i:08d}" for i in range(n_features)]
    signatures = rng.beta(0.8, 0.8, size=(8, n_features))
    cohorts = {}
    for name, count, prefix in [
        ("train", 12, "train"),
        ("test", 4, "test"),
        ("hm450_validation", 3, "paired"),
    ]:
        y = np.repeat(LABELS, count)
        values = np.clip(
            signatures[y.astype(int) - 1] + rng.normal(0, 0.14, (len(y), n_features)), 0.001, 0.999
        )
        ids = pd.Index([f"synthetic-{prefix}-{i:03d}" for i in range(len(y))])
        cohorts[name] = Cohort(
            pd.DataFrame(values, index=ids, columns=columns),
            pd.Series(y, index=ids, name="subtype"),
        )
    paired = cohorts["hm450_validation"]
    values = np.clip(paired.X.to_numpy() + rng.normal(0.015, 0.04, paired.X.shape), 0.001, 0.999)
    cohorts["epic_validation"] = Cohort(
        pd.DataFrame(values, index=paired.X.index, columns=columns), paired.y.copy()
    )
    dataset = Dataset(
        cohorts,
        _summary(
            cohorts,
            [
                "Artificial subtype signatures; "
                "scores do not measure biological or clinical performance."
            ],
        ),
        "demo",
    )
    validate_dataset(dataset)
    return dataset
