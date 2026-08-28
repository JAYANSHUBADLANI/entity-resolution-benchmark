from src.er.blocking import evaluate_blocking, exact_field, full_cartesian, tfidf_neighbours


def test_full_cartesian_keeps_everything(tiny_benchmark):
    result = evaluate_blocking("all", full_cartesian(tiny_benchmark), tiny_benchmark)
    assert result["candidate_pairs"] == 16
    assert result["pairs_completeness"] == 1.0
    assert result["reduction_ratio"] == 0.0


def test_exact_title_blocking_loses_the_inexact_matches(tiny_benchmark):
    pairs = exact_field(tiny_benchmark, "title")
    result = evaluate_blocking("exact title", pairs, tiny_benchmark)
    # l1-r1 and l3-r3 agree exactly. l2-r2 and l3-r4 do not, so two of four are lost.
    assert result["true_matches_kept"] == 2
    assert result["matches_lost"] == 2


def test_recall_ceiling_equals_pairs_completeness(tiny_benchmark):
    pairs = tfidf_neighbours(tiny_benchmark, "title", k=2)
    result = evaluate_blocking("tfidf", pairs, tiny_benchmark)
    assert result["recall_ceiling"] == result["pairs_completeness"]
    assert result["candidate_pairs"] == 8
