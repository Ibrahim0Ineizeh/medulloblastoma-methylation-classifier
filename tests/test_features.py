import numpy as np
import pytest
from sklearn.base import clone

from mb_classifier.features import TopVarianceSelector


def test_selector_fits_training_variance_only():
    train = np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.2], [0.0, 0.0, 0.3]])
    held_out = np.array([[0.5, 100.0, 0.4]])
    selector = TopVarianceSelector(1).fit(train)
    assert selector.get_support(indices=True).tolist() == [0]
    assert selector.transform(held_out).tolist() == [[0.5]]
    assert clone(selector).get_params() == {"n_features": 1}


def test_stable_tie_and_order():
    selector = TopVarianceSelector(2).fit(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert selector.get_support(indices=True).tolist() == [0, 1]


def test_invalid_probe_count_and_transform_width():
    with pytest.raises(ValueError, match="positive integer"):
        TopVarianceSelector(0).fit([[0, 1], [1, 0]])
    selector = TopVarianceSelector(1).fit([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="feature count"):
        selector.transform([[0, 1, 2]])
