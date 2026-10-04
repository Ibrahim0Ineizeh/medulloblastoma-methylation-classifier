"""Exercise the prepared-data contract with small, deliberately awkward inputs."""

import copy

import numpy as np
import pandas as pd
import pytest

from mb_classifier.data import (
    INPUT_FILES,
    load_prepared_data,
    validate_beta_matrix,
    validate_dataset,
)


@pytest.fixture
def prepared_inputs(tmp_path):
    """Keep all eight labels while varying row order, probe order, and stored types."""
    labels = [str(i) for i in range(1, 9)]
    train_ids = [f"GSM{i:07d}" for i in range(1, 9)]
    test_ids = [f"GSM{i:07d}" for i in range(9, 17)]
    epic_ids = [f"GSM{i:07d}" for i in range(17, 25)]
    all_ids = train_ids + test_ids + epic_ids
    metadata = pd.DataFrame(
        {
            "gsm": all_ids,
            "platform": ["GPL13534"] * 16 + ["GPL21145"] * 8,
            "subtype": labels * 3,
            "subgroup": ["MB_G3"] * 12 + ["MB_G4"] * 12,
        }
    )
    metadata.to_csv(tmp_path / INPUT_FILES[2], index=False)

    probes = ["cg00000003", "cg00000001", "cg00000002", "cg00000099"]
    values = np.linspace(0, 1, len(all_ids) * len(probes)).reshape(len(all_ids), -1)
    hm = pd.DataFrame(values, index=pd.Index(all_ids, name="gsm"), columns=probes)
    hm["subtype"] = labels * 3
    hm["sample_metadata"] = "must never be a feature"
    hm = hm.iloc[::-1]
    hm.to_parquet(tmp_path / INPUT_FILES[0])

    epic = hm.loc[epic_ids, ["cg00000002", "cg00000003", "cg00000001"]].copy()
    epic["cg00000098"] = 0.25
    # Real EPIC prepared files store labels numerically, including float labels.
    epic["subtype"] = np.arange(1, 9, dtype=float)
    epic["sample_metadata"] = "different irrelevant metadata"
    epic = epic.iloc[::-1]
    epic.to_parquet(tmp_path / INPUT_FILES[1])

    # Preserve a manifest order different from the matrix and lexical ID order.
    manifest_ids = [train_ids[i] for i in [3, 0, 7, 1, 5, 2, 6, 4]]
    manifest = pd.DataFrame(
        {
            "Samples": manifest_ids,
            "Group": metadata.set_index("gsm").loc[manifest_ids, "subtype"].to_numpy(),
            "cg00000001": 999.0,
            "unrelated_panel_column": "malicious label-like content",
        }
    )
    manifest.to_csv(tmp_path / INPUT_FILES[3], index=False)
    return {
        "path": tmp_path,
        "hm": hm,
        "epic": epic,
        "metadata": metadata,
        "manifest": manifest,
        "train_ids": manifest_ids,
        "test_ids": test_ids,
        "epic_ids": epic_ids,
    }


def test_prepared_loader_uses_only_common_cpgs_and_preserves_manifest_ids(prepared_inputs):
    inputs = prepared_inputs
    dataset = load_prepared_data(inputs["path"])
    common = ["cg00000001", "cg00000002", "cg00000003"]
    train = dataset.cohorts["train"]

    assert train.X.index.tolist() == inputs["train_ids"]
    assert dataset.cohorts["test"].X.index.tolist() == inputs["test_ids"]
    assert dataset.summary["n_features"] == 3
    assert dataset.summary["probe_alignment"] == {
        "common_cpgs": 3,
        "representations": {
            "hm450": {"input_cpgs": 4, "excluded_cpgs": 1},
            "epic": {"input_cpgs": 4, "excluded_cpgs": 1},
        },
    }
    assert dataset.mode == "real"
    for name, cohort in dataset.cohorts.items():
        assert cohort.X.columns.tolist() == common
        assert cohort.X.index.equals(cohort.y.index)
        assert set(cohort.y) == set("12345678")
        source = inputs["epic"] if name == "epic_validation" else inputs["hm"]
        pd.testing.assert_frame_equal(
            cohort.X,
            source.loc[cohort.X.index, common].astype(float),
            check_names=False,
        )
    # The existing supervised panel's values must not enter the feature matrix.
    assert (train.X["cg00000001"] != 999).all()


def test_all_epic_metadata_ids_remain_reserved_when_a_measurement_is_missing(prepared_inputs):
    inputs = prepared_inputs
    missing_id = inputs["epic_ids"][3]
    inputs["epic"].drop(index=missing_id).to_parquet(inputs["path"] / INPUT_FILES[1])
    dataset = load_prepared_data(inputs["path"])

    assert len(dataset.cohorts["hm450_validation"].X) == 8
    assert len(dataset.cohorts["epic_validation"].X) == 7
    assert missing_id in dataset.cohorts["hm450_validation"].X.index
    assert missing_id not in dataset.cohorts["test"].X.index
    assert missing_id not in dataset.cohorts["train"].X.index

    extra_row = pd.DataFrame({"Samples": [missing_id], "Group": ["4"]})
    pd.concat([inputs["manifest"], extra_row], ignore_index=True).to_csv(
        inputs["path"] / INPUT_FILES[3], index=False
    )
    with pytest.raises(ValueError, match="includes EPIC validation specimens"):
        load_prepared_data(inputs["path"])


@pytest.mark.parametrize("holdout", ["test", "hm450_validation", "epic_validation"])
def test_training_overlap_is_rejected(prepared_inputs, holdout):
    dataset = copy.deepcopy(load_prepared_data(prepared_inputs["path"]))
    train, other = dataset.cohorts["train"], dataset.cohorts[holdout]
    train.X = pd.concat([train.X, other.X.iloc[[0]]])
    train.y = pd.concat([train.y, other.y.iloc[[0]]])

    with pytest.raises(ValueError, match="Training specimens overlap"):
        validate_dataset(dataset)


@pytest.mark.parametrize("file_index", [0, 1, 2, 3])
def test_duplicate_specimen_ids_are_rejected(prepared_inputs, file_index):
    inputs = prepared_inputs
    path = inputs["path"] / INPUT_FILES[file_index]
    if file_index < 2:
        frame = pd.read_parquet(path)
        pd.concat([frame, frame.iloc[[0]]]).to_parquet(path)
    else:
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(ValueError, match="[Dd]uplicate|unique"):
        load_prepared_data(inputs["path"])


@pytest.mark.parametrize("bad_value", [-0.001, 1.001, np.nan, np.inf, -np.inf])
def test_invalid_beta_values_are_rejected(prepared_inputs, bad_value):
    inputs = prepared_inputs
    frame = inputs["hm"].copy()
    frame.loc[inputs["train_ids"][0], "cg00000001"] = bad_value
    frame.to_parquet(inputs["path"] / INPUT_FILES[0])

    with pytest.raises(ValueError, match="finite|within"):
        load_prepared_data(inputs["path"])


def test_beta_contract_rejects_duplicate_probes_and_metadata_features():
    duplicate_probes = pd.DataFrame([[0.1, 0.2]], columns=["cg1", "cg1"])
    with pytest.raises(ValueError, match="unique"):
        validate_beta_matrix(duplicate_probes)
    target_feature = pd.DataFrame({"cg1": [0.1], "subtype": [1]})
    with pytest.raises(ValueError, match="Every feature must be a CpG"):
        validate_beta_matrix(target_feature)


@pytest.mark.parametrize("missing_file", INPUT_FILES)
def test_missing_input_names_the_file_and_documents_remedy(prepared_inputs, missing_file):
    inputs = prepared_inputs
    (inputs["path"] / missing_file).unlink()

    with pytest.raises(FileNotFoundError) as error:
        load_prepared_data(inputs["path"])
    assert missing_file in str(error.value)
    assert "docs/DATA.md" in str(error.value)
    assert "demo" in str(error.value)


@pytest.mark.parametrize("file_index", [0, 1])
def test_embedded_labels_must_match_metadata(prepared_inputs, file_index):
    inputs = prepared_inputs
    path = inputs["path"] / INPUT_FILES[file_index]
    frame = pd.read_parquet(path)
    # Use a valid, different subtype so the failure checks agreement, not format.
    original = str(int(float(frame.iloc[0]["subtype"])))
    wrong = "1" if original != "1" else "2"
    frame["subtype"] = frame["subtype"].astype(str)
    frame.loc[frame.index[0], "subtype"] = wrong
    frame.to_parquet(path)

    with pytest.raises(ValueError, match="Embedded subtype disagrees"):
        load_prepared_data(inputs["path"])


def test_manifest_labels_must_match_metadata(prepared_inputs):
    inputs = prepared_inputs
    manifest = inputs["manifest"].copy()
    manifest.loc[0, "Group"] = "1"
    manifest.to_csv(inputs["path"] / INPUT_FILES[3], index=False)

    with pytest.raises(ValueError, match="manifest Group labels disagree"):
        load_prepared_data(inputs["path"])
