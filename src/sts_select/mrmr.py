from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator

from .scoring import BaseScorer


class MRMRBase(BaseEstimator):
    def __init__(
        self,
        scorer: BaseScorer,
        n_features=30,
    ):
        self.n_features = n_features
        self.scorer = scorer
        self.sel_features = None
        self.relevancies = None
        self.redundancies = None
        self.mrmr = None

        if not isinstance(scorer, BaseScorer):
            raise TypeError("scorer must be a BaseScorer")

        if not self.scorer.scored():
            raise ValueError("Must have redundancy/relevancy scores before running mRMR.")

    def distance_X_y(self, original):
        """
        Averages X-y score pairs for a given feature.
        :param original:
        :return:
        """
        dist = np.mean(
            [self.scorer.X_y_score(original, i) for i in range(self.scorer.y_n)]
        )
        return dist

    def score_label(self, vec, label):
        """
        Returns the score for each feature and the overall label.
        :param vec:
        :param label:
        :return:
        """
        return [self.distance_X_y(x) for x in range(self.scorer.X_n)]

    def score_lfs_candidate(self, X, candidates, lastFeatureSelected):
        """
        Returns the score between candidates and the last feature selected.
        :param X:
        :param candidates:
        :param lastFeatureSelected:
        :return:
        """
        cs = [0] * len(candidates)
        for cidx, c in enumerate(candidates):
            cs[cidx] = self.scorer.X_score(cidx, lastFeatureSelected)
        return cs

    def transform(self, X):
        return X[:, self.sel_features]

    def fit(self, X, y):
        # Adapted from fast-mRMR without optimizations (Ramírez-Gallego et al.)

        selectedFeatures = list()
        candidates = set(x for x in range(X.shape[1]))
        candidatesVec = [True for x in range(X.shape[1])]
        accumulatedRedundancy = defaultdict(float)  # No argmax here

        # Double check: each feature is a column right?
        relevancesVector = self.score_label(X, y)

        selected = int(np.argmax(relevancesVector))
        lastFeatureSelected = selected
        selectedFeatures.append(
            (selected, relevancesVector[selected], relevancesVector[selected], 0)
        )
        candidates.remove(selected)

        while len(selectedFeatures) < self.n_features:
            max_mrmr = -np.inf
            max_rel = None
            max_red = None
            newLastFeatureSelected = None

            lastFeatureSelectedMI = self.score_lfs_candidate(X, candidatesVec, lastFeatureSelected)
            for idxc, can in enumerate(candidates):
                relevance = relevancesVector[can]
                accumulatedRedundancy[can] += lastFeatureSelectedMI[idxc]
                redundancy = accumulatedRedundancy[can] / len(selectedFeatures)
                mrmr = relevance - redundancy
                if mrmr > max_mrmr:
                    max_mrmr = float(mrmr)
                    max_rel = float(relevance)
                    max_red = float(redundancy)
                    newLastFeatureSelected = can

            selectedFeatures.append(
                (newLastFeatureSelected, max_mrmr, max_rel, max_red)
            )
            candidates.remove(newLastFeatureSelected)
            candidatesVec[newLastFeatureSelected] = False
            lastFeatureSelected = newLastFeatureSelected

        self.sel_features, self.mrmr, self.relevancies, self.redundancies = [
            list(x) for x in zip(*selectedFeatures)
        ]
        # Assign each of the selected features and their scores to sel_fetures, mrmr, relevancies, redundancies
        return self

