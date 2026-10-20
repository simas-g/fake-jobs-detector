"""k-nearest neighbours classifier written from scratch (numpy only)."""
import numpy as np


class KNN:
    def __init__(self, k=3, scaling="standard"):
        self.k = k
        self.scaling = scaling  # "standard", "minmax" or None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if self.scaling == "standard":
            self.center_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
        elif self.scaling == "minmax":
            self.center_ = X.min(axis=0)
            self.scale_ = X.max(axis=0) - X.min(axis=0)
        else:
            self.center_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1.0  # constant column: leave it unscaled
        self.X_ = self._scale(X)
        self.y_ = np.asarray(y, dtype=int)
        return self

    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.center_) / self.scale_

    def kneighbors(self, X):
        """Indices (into the training set) and Euclidean distances of the k nearest."""
        Xs = self._scale(np.atleast_2d(X))
        dist = np.sqrt(((Xs[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=2))
        idx = np.argsort(dist, axis=1, kind="stable")[:, : self.k]
        return idx, np.take_along_axis(dist, idx, axis=1)

    def predict_proba(self, X):
        """Share of the k neighbours that are scams."""
        idx, _ = self.kneighbors(X)
        return self.y_[idx].mean(axis=1)

    def predict(self, X):
        # majority vote; with odd k there are no ties
        return (self.predict_proba(X) > 0.5).astype(int)
