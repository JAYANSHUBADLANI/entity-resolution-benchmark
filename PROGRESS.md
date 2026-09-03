# Progress

## Done

**Phase 1, data and profiling.** Both benchmarks downloaded from the Leipzig database group with
the archive hash and retrieval date recorded in `data/raw/provenance.json`. Ground truth
reconciled against the sources: no orphan labels, no duplicate ids, so the labels are usable as
they stand. Report in `reports/01_data_profile.md`.

Facts the profiling turned up that shaped everything after it:

- DBLP-ACM is strictly one to one. Amazon-GoogleProducts is not: one Amazon product matches up to
  5 Google listings, and 144 Amazon records have more than one match. Any 1:1 assignment step
  would cap recall on the second dataset, so it cannot be applied blindly to both.
- `manufacturer` is missing on 92.8 percent of Google records. It looks like an obvious blocking
  key and is useless as one.
- `description` averages 1,283 characters on Amazon and 195 on Google. The same field name means
  different things on the two sides.

**Phase 2, blocking.** Eight schemes measured on pairs completeness against reduction ratio,
`reports/02_blocking.md`. Headline numbers:

- Exact match on `venue` in DBLP-ACM returns **zero** candidate pairs. The venue is abbreviated on
  one side and spelled out on the other, so a blocking key that reads as obviously safe silently
  deletes the entire problem.
- Exact match on `title` retains 91.1 percent of DBLP-ACM matches but only 3.9 percent of
  Amazon-Google matches.
- TF-IDF character n-gram nearest neighbours at k=20 keeps 100 percent of DBLP-ACM matches and
  98.9 percent of Amazon-Google matches while scoring 0.87 and 0.62 percent of the pair space.
  This is the candidate set used everywhere downstream.
- On the dirty dataset the recall ceiling never reaches 1.0 short of near cartesian: k=50 gets
  99.4 percent, shared-token blocking 99.7 percent at 10.9 percent of the space.

**Phase 3, matching.** Four matchers, five fold cross validation, two splitting regimes,
`artifacts/matching_results.csv`. Best end to end F1: **0.986 on DBLP-ACM** (gradient boosting)
and **0.581 on Amazon-Google** (gradient boosting). Fellegi-Sunter, which uses no labels at all,
reaches 0.90 on DBLP-ACM once its inputs are fixed, see below.

**Phase 3b, two follow up experiments** prompted by results that did not match expectations.

*The split experiment came out negative.* Splitting candidate pairs at random rather than by
record was expected to inflate F1. It does not: the gap is between -0.11 and +0.29 percentage
points across every matcher and both datasets, which is noise. The reason is that every feature
here is a symmetric similarity function, so no feature carries the identity of a particular
record and there is nothing for the model to memorise.

That explanation was then tested rather than asserted. Adding one feature that does carry record
identity, a target encoding of the left record id, changes the picture completely, and not in the
direction expected either: under a pair level split F1 **collapses** from 0.988 to 0.405 on
DBLP-ACM and from 0.584 to 0.207 on Amazon-Google, while under an entity level split it is
unchanged. Verified mechanism, measured on one fold: the encoded feature correlates **+0.143 with
the label in training and -0.277 in test**. Because each DBLP record has exactly one true match,
a test positive's record contributes only negatives to the training fold, so its encoding is
exactly 0.0000 while training positives sit at 0.0625. The feature's meaning inverts between fit
and prediction. Under an entity level split the same feature is constant at the prior for unseen
records, correlation exactly 0.0000, and the model ignores it.

So the useful statement is not "pair level splits inflate scores". It is that a pair level split
cannot detect an identity carrying feature, and the failure it hides can be catastrophic in
either direction.

*Fellegi-Sunter's weak precision is explained by its own assumption.* Fitted on all 17 features it
scores 0.765 F1 on DBLP-ACM at 0.619 precision. Fitted on one feature per field, four features, it
reaches **0.901 F1 at 0.820 precision** with recall unchanged. Four correlated views of the same
title were being counted as four independent pieces of evidence, exactly what conditional
independence forbids. On Amazon-Google the same change moves F1 only 0.448 to 0.462 and EM hits
the 200 iteration cap without converging, which is reported as it stands rather than tuned away.

## Pending

Nothing in the build is outstanding. Every phase runs from `make demo` in about two and a half
minutes, `make test` passes 17 tests with no network, and `make determinism` confirms two
consecutive runs produce byte identical artifacts.

Remaining work:

- Optional: a third, harder benchmark (Walmart-Amazon or Abt-Buy) would test whether the
  conclusions here are about the method or about these two datasets. The loader is already
  config driven, so adding one is a config entry plus a download.
- Optional: a transformer baseline on the dirty dataset, purely to see how much of the gap to the
  published deep result is method and how much is candidate set.

## Decisions for me

- Whether to add the third benchmark before pushing, or push as is.
- Repo name is currently `entity-resolution-benchmark`.
