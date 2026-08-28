import numpy as np
import pandas as pd

from src.er.resolve import (constraint_ceiling, degree_capped_greedy, hungarian_one_to_one,
                            observed_degrees)


def test_one_to_one_ceiling_counts_what_it_deletes(tiny_benchmark):
    result = constraint_ceiling(tiny_benchmark.matches, 1, 1)
    # l3 has two true matches, so a strict 1:1 rule can keep only one of them.
    assert result["keepable_matches"] == 3
    assert result["total_matches"] == 4
    assert result["recall_ceiling"] == 0.75


def test_observed_degrees_reads_the_real_cardinality(tiny_benchmark):
    assert observed_degrees(tiny_benchmark.matches) == (2, 1)


def test_degree_cap_is_respected_and_prefers_high_scores():
    pairs = pd.DataFrame({"left_id": ["a", "a", "b"], "right_id": ["x", "y", "x"]})
    score = np.array([0.9, 0.8, 0.7])
    decision = degree_capped_greedy(pairs, score, threshold=0.5, max_left=1, max_right=1)
    # a-x wins on score, which then blocks both a-y (a is used) and b-x (x is used).
    assert list(decision) == [1, 0, 0]

    relaxed = degree_capped_greedy(pairs, score, threshold=0.5, max_left=2, max_right=1)
    assert list(relaxed) == [1, 1, 0]


def test_threshold_excludes_low_scores():
    pairs = pd.DataFrame({"left_id": ["a"], "right_id": ["x"]})
    assert list(degree_capped_greedy(pairs, np.array([0.3]), 0.5, 1, 1)) == [0]


def test_hungarian_matches_greedy_when_greedy_is_optimal():
    pairs = pd.DataFrame({"left_id": ["a", "a", "b"], "right_id": ["x", "y", "x"]})
    score = np.array([0.9, 0.8, 0.7])
    greedy = degree_capped_greedy(pairs, score, 0.5, 1, 1)
    exact = hungarian_one_to_one(pairs, score, 0.5)
    # Greedy takes a-x and stops. The assignment solver takes a-y and b-x, total 1.5 against 0.9.
    assert greedy.sum() == 1
    assert exact.sum() == 2
