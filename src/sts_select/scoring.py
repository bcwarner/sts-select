import inspect
import itertools
import os
import pickle
import re
import warnings
from collections import defaultdict
from typing import Dict, Tuple

import tqdm
import torch


class BaseScorer:
    def __init__(
        self, X, y, X_names=None, y_names=None, cache=None, verbose=0, **kwargs
    ):
        """
        Base scorer class. All scorers should inherit from this class.
        :param X:
        :param y:
        :param X_names:
        :param y_names:
        :param test:
        :param kwargs:
        :param cache:
        """
        self.X = X
        self.y = y
        self.X_names = list() if X_names is None else X_names
        self.y_names = list() if y_names is None else y_names
        self.cache = cache
        self.verbose = verbose
        self.__dict__.update(kwargs)

        self.X_n = X.shape[1]
        # Assumption: y_names and y will never be scored together
        self.y_n = max(len(self.y_names), 1)

        # Value of different pairings of X and y
        self.X_pairings: Dict[Tuple[int, int], float] = defaultdict(float)
        self.X_y_pairings: Dict[Tuple[int, int], float] = defaultdict(float)
        if self.cache and os.path.exists(self.cache):
            self.load_cache()
        else:
            self.score(self.X, self.y)

    def scored(self):
        """
        Check if the scorer has already been initialized.
        :return:
        """
        return len(self.X_pairings) > 0 and len(self.X_y_pairings) > 0

    def _X_score(self, x1, x2):
        raise NotImplementedError()

    def _X_y_score(self, x, y):
        raise NotImplementedError()

    def score(self, X, y):
        """
        Score the given X and y.
        :param X:
        :param y:
        :return:
        """
        iterator = itertools.combinations(range(X.shape[1]), 2)
        if self.verbose > 0:
            iterator = tqdm.tqdm(
                iterator,
                total=X.shape[1] * (X.shape[1] - 1) / 2,
                desc=f"Scoring {self.__class__.__name__}",
            )

        for x1, x2 in iterator:
            self.X_pairings[(x1, x2)] = self._X_score(x1, x2)

        # Assumption: y_names and y will never be scored together
        iterator = itertools.product(
            range(X.shape[1]), range(max(len(self.y_names), 1))
        )
        if self.verbose > 0:
            iterator = tqdm.tqdm(
                iterator,
                total=X.shape[1] * max(len(self.y_names), 1),
                desc=f"Scoring {self.__class__.__name__}",
            )
        for x, y in iterator:
            self.X_y_pairings[(x, y)] = self._X_y_score(x, y)

        if self.cache:
            self.save_cache()

    def X_score(self, x1: int, x2: int):
        # Ensure that x1 < x2
        x1, x2 = min(x1, x2), max(x1, x2)
        return self.X_pairings[(x1, x2)]

    def X_y_score(self, x: int, y: int):
        return self.X_y_pairings[(x, y)]

    def load_cache(self):
        """
        Load the cache with the given name.
        :param name:
        :return:
        """
        #

        if self.cache is None:
            raise ValueError("Cache is None")

        with open(os.path.join(self.cache), "rb") as f:
            dat = pickle.load(f)
            if self.X_n != dat["X_n"]:
                raise ValueError(
                    f"Feature count does not match cache {self.X.shape[1]} != {dat['X_n']}"
                )
            if self.y_n != dat["y_n"]:
                raise ValueError(
                    f"Label count does not match cache {self.y_n} != {dat['y_n']}"
                )

            self.X_pairings = dat["X_pairings"]
            self.X_y_pairings = dat["X_y_pairings"]
            self.X_n = dat["X_n"]
            self.y_n = dat["y_n"]

    def save_cache(self):
        """
        Save the cache with the given name.
        :param name:
        :return:
        """
        if self.cache is None:
            raise ValueError("Cache is None")

        with open(os.path.join(self.cache), "wb") as f:
            out_dict = {
                "X_pairings": self.X_pairings,
                "X_y_pairings": self.X_y_pairings,
                "X_n": self.X_n,
                "y_n": self.y_n,
            }
            pickle.dump(out_dict, f)


class LinearScorer(BaseScorer):
    def __init__(self, X, y, cache=None, **kwargs):
        super().__init__(X, y, cache=cache, **kwargs)
        self.scorers = kwargs["scorers"]
        self.alpha = kwargs["alpha"]

    def _X_score(self, x1, x2):
        # Linear combination of the alpha weights and the results from the individual scorers.
        return sum(
            [
                self.alpha[i] * self.scorers[i].X_score(x1, x2)
                for i in range(len(self.scorers))
            ]
        )

    def _X_y_score(self, x, y):
        # Linear combination of the alpha weights and the results from the individual scorers.
        return sum(
            [
                self.alpha[i] * self.scorers[i].X_y_score(x, y)
                for i in range(len(self.scorers))
            ]
        )

    def set_params(self, **params):
        # Hack to make this work with array values. Should be fixed to work with sklearn correctly.
        for pk, pv in params.items():
            match = re.match(r"(.*)\[(\d+)\]", pk)
            if match:
                pk, idx = match.groups()
                idx = int(idx)
                getattr(self, pk)[idx] = pv
            else:
                setattr(self, pk, pv)


class MIScorer(BaseScorer):
    """
    Scorer for mutual information.
    """

    def __init__(self, X, y, cache=None, random_state=0, **kwargs):
        self.random_state = random_state
        super().__init__(X, y, cache=cache, **kwargs)

    def _X_score(self, x1, x2):
        from sklearn.feature_selection import mutual_info_regression

        return mutual_info_regression(
            self.X[:, x1].reshape(-1, 1),
            self.X[:, x2],
            discrete_features=False,
            random_state=self.random_state,
        ).item()

    def _X_y_score(self, x, y):
        from sklearn.feature_selection import mutual_info_classif

        return mutual_info_classif(
            self.X[:, x].reshape(-1, 1),
            self.y,
            discrete_features=False,
            random_state=self.random_state,
        ).item()


class BaseSTSScorer(BaseScorer):
    def __init__(self, X, y, X_names=None, y_names=None, cache=None, **kwargs):
        """
        Base class for using semantic textual similairty to score.
        Assumes that we have a simple function that takes two strings and returns a score on a uniform scale.
        :param X:
        :param y:
        :param X_names:
        :param y_names:
        :param cache:
        :param kwargs:
        """
        super().__init__(X, y, X_names=X_names, y_names=y_names, cache=cache, **kwargs)
        self.sts_function = kwargs["sts_function"]

        # Check that the sts_function is provided and accepts two string parameters (do not test type)
        if self.sts_function is None:
            # Display warning, but don't except
            warnings.warn("sts_function is None")
        else:
            if not callable(self.sts_function):
                raise ValueError(f"sts_function {self.sts_function} is not callable")

            if len(inspect.signature(self.sts_function).parameters) != 2:
                raise ValueError(
                    f"sts_function {self.sts_function} should accept two parameters"
                )

    def _X_score(self, x1, x2):
        return self.sts_function(self.X_names[x1], self.X_names[x2])

    def _X_y_score(self, x, y):
        return self.sts_function(self.X_names[x], self.y_names[y])


class SentenceTransformerScorer(BaseSTSScorer):
    def __init__(
        self,
        X,
        y,
        X_names=None,
        y_names=None,
        cache=None,
        model_path=None,
        verbose=0,
        **kwargs,
    ):
        if model_path is None:
            raise ValueError("model_path must be specified.")
        self.model_path = model_path

        super().__init__(
            X,
            y,
            X_names=X_names,
            y_names=y_names,
            cache=cache,
            sts_function=lambda a, b: None,
            verbose=verbose,
            **kwargs,
        )

    def score(self, X, y):
        if self.sts_function("", "") is None:
            from sentence_transformers import SentenceTransformer, util

            self.model = SentenceTransformer(self.model_path,
                                             device="cuda" if torch.cuda.is_available() else "cpu")
            self.sts_function = lambda x1, x2: util.pytorch_cos_sim(
                self.model.encode(x1, show_progress_bar=self.verbose > 2),
                self.model.encode(x2, show_progress_bar=self.verbose > 2)
            ).item()
        return super().score(X, y)


class GensimScorer(BaseSTSScorer):
    def __init__(
        self,
        X,
        y,
        X_names=None,
        y_names=None,
        cache=None,
        model_path=None,
        verbose=0,
        model_type: type = None,
        **kwargs,
    ):
        if model_path is None:
            raise ValueError("model_path must be specified.")
        self.model_path = model_path
        if model_type is None:
            raise ValueError("model_type must be specified.")
        self.model_type = model_type

        super().__init__(
            X,
            y,
            X_names=X_names,
            y_names=y_names,
            cache=cache,
            sts_function=lambda a, b: None,
            verbose=verbose,
            **kwargs,
        )

    def score(self, X, y):
        if self.sts_function("", "") is None:
            from sentence_transformers import util

            self.model = self.model_type(self.model_path)
            self.sts_function = lambda x1, x2: util.pytorch_cos_sim(
                torch.sum(torch.Tensor(self.model.encode(x1)), dim=0),
                torch.sum(torch.Tensor(self.model.encode(x2)), dim=0),
            ).item()
        return super().score(X, y)
