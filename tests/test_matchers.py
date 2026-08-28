import numpy as np
import pandas as pd

from src.er.matchers import FellegiSunter, ThresholdMatcher, f1, score_predictions


def test_f1_arithmetic():
    y = np.array([1, 1, 0, 0])
    assert f1(y, np.array([1, 1, 0, 0])) == 1.0
    assert f1(y, np.array([0, 0, 0, 0])) == 0.0
    # One true positive, one false positive, one false negative: precision and recall both 0.5.
    assert round(f1(y, np.array([1, 0, 1, 0])), 4) == 0.5


def test_end_to_end_recall_uses_the_wider_denominator():
    y = np.array([1, 0])
    pred = np.array([1, 0])
    result = score_predictions(y, pred, truth_total=4)
    assert result["recall_in_candidates"] == 1.0
    assert result["recall_end_to_end"] == 0.25
    assert result["f1_end_to_end"] < result["f1_in_candidates"]


def test_threshold_matcher_finds_the_informative_feature():
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 50)
    X = pd.DataFrame({
        "useful": np.concatenate([rng.uniform(0.8, 1.0, 50), rng.uniform(0.0, 0.2, 50)]),
        "noise": rng.uniform(0, 1, 100),
    })
    model = ThresholdMatcher().fit(X, y)
    assert model.feature_ == "useful"
    assert f1(y, model.predict(X)) > 0.95


def test_fellegi_sunter_separates_a_clean_mixture():
    rng = np.random.default_rng(1)
    matches = rng.uniform(0.8, 1.0, (100, 3))
    non_matches = rng.uniform(0.0, 0.3, (400, 3))
    X = pd.DataFrame(np.vstack([matches, non_matches]), columns=list("abc"))
    y = np.array([1] * 100 + [0] * 400)

    model = FellegiSunter(seed=0).fit(X)
    posterior = model.posterior(X)
    assert posterior[y == 1].mean() > posterior[y == 0].mean()
    assert f1(y, model.predict(X)) > 0.9
    # m is the chance of agreement among matches, u among non matches. m must dominate.
    assert (model.m_ > model.u_).all()
