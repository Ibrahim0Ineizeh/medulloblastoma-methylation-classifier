"""Training-only selection of variable CpG probes."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class TopVarianceSelector(TransformerMixin, BaseEstimator):
    """Select variable probes without treating nominal subtype codes as numbers.

    Stable sorting makes equal-variance ties deterministic. Fitting happens
    inside the model pipeline, independently in each cross-validation fold.
    """

    def __init__(self, n_features=1000):
        self.n_features = n_features

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64)
        if not isinstance(self.n_features, int) or self.n_features < 1:
            raise ValueError("n_features must be a positive integer")
        self.n_features_in_ = X.shape[1]
        variance = np.var(X, axis=0)
        order = np.argsort(-variance, kind="stable")
        self.indices_ = np.sort(order[: min(self.n_features, X.shape[1])])
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count differs from the training matrix")
        return X[:, self.indices_]

    def get_support(self, indices=False):
        check_is_fitted(self, "indices_")
        if indices:
            return self.indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.indices_] = True
        return mask
