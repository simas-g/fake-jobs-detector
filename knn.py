"""k-nearest neighbours: scale the features, Euclidean distance, majority vote."""
import numpy as np


class KNN:
    def __init__(self, k=3, scaling="standard"):
        self.k = k
        self.scaling = scaling  # "standard", "minmax" or None

    def fit(self, X, y):
        if self.scaling == "standard":
            self.shift, self.scale = X.mean(axis=0), X.std(axis=0)
        elif self.scaling == "minmax":
            self.shift, self.scale = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
        else:
            self.shift, self.scale = 0, 1
        self.scale = np.where(self.scale == 0, 1, self.scale)  # avoid dividing by 0

        self.X = (X - self.shift) / self.scale
        self.y = np.asarray(y)
        return self

    def kneighbors(self, X):
        """Row numbers and distances of the k closest training postings."""
        X = (X - self.shift) / self.scale
        distances = np.sqrt(((X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2))
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :self.k]
        return nearest, np.take_along_axis(distances, nearest, axis=1)

    def predict(self, X):
        nearest, _ = self.kneighbors(X)
        votes = self.y[nearest].sum(axis=1)
        return (votes > self.k / 2).astype(int)
