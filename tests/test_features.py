import math

import pandas as pd

from src.er.features import build_features, containment, jaccard, label_pairs


def test_jaccard_and_containment_edge_cases():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert math.isnan(jaccard(set(), set()))
    # Containment rewards a subset where jaccard punishes the length difference.
    assert containment({"a"}, {"a", "b", "c"}) == 1.0
    assert jaccard({"a"}, {"a", "b", "c"}) < 0.5


def test_features_and_labels_line_up_with_the_pairs(tiny_benchmark):
    pairs = [("l1", "r1"), ("l1", "r2"), ("l3", "r4")]
    features = build_features(tiny_benchmark, pairs)
    labels = label_pairs(features, tiny_benchmark)

    assert list(zip(features["left_id"], features["right_id"])) == pairs
    assert list(labels) == [1, 0, 1]
    # Identical titles must score 1.0, unrelated ones below it.
    assert features.loc[0, "title_jaccard"] == 1.0
    assert features.loc[1, "title_jaccard"] < 1.0
