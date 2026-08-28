"""Matchers, from the cheapest defensible baseline up to a supervised model.

Three families, deliberately:

- a single similarity with a threshold, which is what a rule based system actually is;
- Fellegi and Sunter's probabilistic model fitted with EM, which needs no labels at all and is
  still the method most production record linkage runs on;
- supervised classifiers, which need labels and are the only ones that can weigh fields against
  each other from evidence.

Reporting the cheap ones is not padding. If the expensive model does not beat them by a margin
worth the labelling effort, that is the finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EPS = 1e-9


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in {"left_id", "right_id"}]


class ThresholdMatcher:
    """Pick one feature and one cut, both chosen on the training fold only."""

    def __init__(self, grid: np.ndarray | None = None):
        self.grid = np.arange(0.05, 1.0, 0.01) if grid is None else grid
        self.feature_ = None
        self.threshold_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        best = (-1.0, None, None)
        for column in X.columns:
            values = X[column].to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            filled = np.nan_to_num(values, nan=0.0)
            for cut in self.grid:
                pred = (filled >= cut).astype(int)
                score = f1(y, pred)
                if score > best[0]:
                    best = (score, column, float(cut))
        _, self.feature_, self.threshold_ = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = np.nan_to_num(X[self.feature_].to_numpy(dtype=float), nan=0.0)
        return (values >= self.threshold_).astype(int)


class FellegiSunter:
    """Classical probabilistic record linkage, fitted with EM. Uses no labels.

    Each similarity is binarised into agree or disagree, then a two component mixture is fitted
    under the conditional independence assumption. That assumption is wrong here (title cosine
    and title jaccard are obviously correlated) and the effect of it is discussed in the report
    rather than hidden, since it is the standard criticism of this model.
    """

    def __init__(self, agree_at: float = 0.6, max_iter: int = 200, tol: float = 1e-8, seed: int = 42):
        self.agree_at = agree_at
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _binarise(self, X: pd.DataFrame) -> np.ndarray:
        values = np.nan_to_num(X.to_numpy(dtype=float), nan=0.0)
        return (values >= self.agree_at).astype(float)

    def fit(self, X: pd.DataFrame, y=None):
        gamma = self._binarise(X)
        n, k = gamma.shape
        rng = np.random.default_rng(self.seed)
        self.p_ = 0.05
        self.m_ = np.clip(rng.uniform(0.7, 0.95, size=k), EPS, 1 - EPS)
        self.u_ = np.clip(rng.uniform(0.05, 0.3, size=k), EPS, 1 - EPS)

        previous = -np.inf
        for _ in range(self.max_iter):
            log_m = gamma @ np.log(self.m_) + (1 - gamma) @ np.log(1 - self.m_)
            log_u = gamma @ np.log(self.u_) + (1 - gamma) @ np.log(1 - self.u_)
            a = np.log(self.p_ + EPS) + log_m
            b = np.log(1 - self.p_ + EPS) + log_u
            top = np.maximum(a, b)
            denom = top + np.log(np.exp(a - top) + np.exp(b - top))
            posterior = np.exp(a - denom)

            self.p_ = float(posterior.mean())
            weight = posterior.sum()
            self.m_ = np.clip((posterior @ gamma) / max(weight, EPS), EPS, 1 - EPS)
            self.u_ = np.clip(((1 - posterior) @ gamma) / max(n - weight, EPS), EPS, 1 - EPS)

            loglik = float(denom.sum())
            if abs(loglik - previous) < self.tol:
                break
            previous = loglik
        self.n_iter_ = _ + 1
        return self

    def posterior(self, X: pd.DataFrame) -> np.ndarray:
        gamma = self._binarise(X)
        a = np.log(self.p_ + EPS) + gamma @ np.log(self.m_) + (1 - gamma) @ np.log(1 - self.m_)
        b = np.log(1 - self.p_ + EPS) + gamma @ np.log(self.u_) + (1 - gamma) @ np.log(1 - self.u_)
        top = np.maximum(a, b)
        denom = top + np.log(np.exp(a - top) + np.exp(b - top))
        return np.exp(a - denom)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.posterior(X) >= 0.5).astype(int)


class SupervisedMatcher:
    def __init__(self, kind: str = "logistic", seed: int = 42):
        self.kind = kind
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if self.kind == "logistic":
            self.scaler_ = StandardScaler()
            values = self.scaler_.fit_transform(np.nan_to_num(X.to_numpy(dtype=float), nan=-1.0))
            self.model_ = LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=self.seed).fit(values, y)
        else:
            self.model_ = HistGradientBoostingClassifier(
                random_state=self.seed, max_iter=300).fit(X.to_numpy(dtype=float), y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "logistic":
            values = self.scaler_.transform(np.nan_to_num(X.to_numpy(dtype=float), nan=-1.0))
            return self.model_.predict_proba(values)[:, 1]
        return self.model_.predict_proba(X.to_numpy(dtype=float))[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, truth_total: int) -> dict:
    """Precision and recall inside the candidate set, plus end to end recall.

    `truth_total` is the number of real matches for the records in this fold, including any that
    blocking never generated a candidate pair for. Recall measured only against candidates that
    survived blocking is the number most write ups quote, and it overstates the system.
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall_in_candidates = tp / (tp + fn) if (tp + fn) else 0.0
    recall_end_to_end = tp / truth_total if truth_total else 0.0
    def harmonic(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall_in_candidates": round(recall_in_candidates, 4),
        "f1_in_candidates": round(harmonic(precision, recall_in_candidates), 4),
        "recall_end_to_end": round(recall_end_to_end, 4),
        "f1_end_to_end": round(harmonic(precision, recall_end_to_end), 4),
    }
