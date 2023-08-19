# Simple use of the STS scores to select features.
import warnings

import numpy as np

from .scoring import BaseScorer


class StdDevSelector:
    """
    Selects features based on the standard deviation of the score from the dataset max.
    From Lampos et al. (2017)
    """
    def __init__(self,
                 scorer: BaseScorer,
                 std_dev: float = 1.0):
        self.scorer = scorer
        self.std_dev = std_dev
        self.sel_features = None

    def fit(self, X, y):
        """
        Select features based on the standard deviation of the score from the dataset max.
        :param X:
        :param y:
        :return:
        """
        # Calculate the distribution of the X-y scores over the range of X averaged for y.
        scores = np.zeros(self.scorer.X_n)
        for x in range(self.scorer.X_n):
            scores[x] = np.mean([self.scorer.X_y_score(x, yi) for yi in range(self.scorer.y_n)])
        mean = np.mean(scores)
        std = np.std(scores)
        # Select features that are above the mean + std_dev * std threadhold'
        min_score = mean + self.std_dev * std
        if min_score > np.max(scores):
            warnings.warn(f"The selected threshold is higher than the maximum score calculated. Selecting max feature.")
            min_score = np.max(scores)
        self.sel_features = [x for x in range(self.scorer.X_n) if scores[x] >= min_score]
        # Sort for readability
        self.sel_features = [x for x in sorted(self.sel_features, key=lambda x: scores[x], reverse=True)]
        self.scores = scores[self.sel_features]
        return self

    def transform(self, X):
        return X[:, self.sel_features]

    def set_params(self, **params):
        # Hack to make this work with array values. Should be fixed to work with sklearn correctly.
        for pk, pv in params.items():
            setattr(self, pk, pv)

class TopNSelector:
    def __init__(self,
                 scorer: BaseScorer,
                 n_features: int = 30):
        self.scorer = scorer
        self.n_features = n_features
        self.sel_features = None

    def fit(self, X, y):
        """
        Select the top N features based on the X-y score.
        :param X:
        :param y:
        :return:
        """
        # Calculate the distribution of the X-y scores over the range of X averaged for y.
        scores = np.zeros(self.scorer.X_n)
        for x in range(self.scorer.X_n):
            scores[x] = np.mean([self.scorer.X_y_score(x, yi) for yi in range(self.scorer.y_n)])
        self.sel_features = [x for x in reversed(np.argsort(scores)[-self.n_features:])]
        self.scores = scores[self.sel_features]
        return self

    def transform(self, X):
        return X[:, self.sel_features]

    def set_params(self, **params):
        # Hack to make this work with array values. Should be fixed to work with sklearn correctly.
        for pk, pv in params.items():
            setattr(self, pk, pv)