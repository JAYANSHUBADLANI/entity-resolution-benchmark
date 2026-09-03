# Entity resolution on two benchmarks

[![tests](https://github.com/JAYANSHUBADLANI/entity-resolution-benchmark/actions/workflows/pytest.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/entity-resolution-benchmark/actions/workflows/pytest.yml)

Deciding which records in two different catalogues describe the same real thing, on two datasets
chosen because they fail in different ways: a clean bibliographic pair where the hard part is
volume, and a dirty product pair where the hard part is that the same item is written up two
completely different ways by two different sellers.

The system is classical throughout: blocking, similarity features, a probabilistic model and a
gradient boosted classifier, then a decision layer. No embeddings, no pretrained language model.
That is a deliberate choice, because the published deep learning results on these exact datasets
are available to compare against, and the interesting question is where the cheap methods hold up
and where they fall over.

## Read this first

Two things about the numbers below, both of which change how they should be read.

**Recall here is measured against the full ground truth, including matches that blocking never
proposed.** It is common to report recall against the candidate set that survived blocking, which
quietly removes the blocking stage from the measurement. Both are reported here and they are not
the same number.

**The comparison with published results is not like for like, and it is not a claim to have beaten
anything.** The published figures are measured on a supplied set of labelled candidate pairs with
a far friendlier class balance than the candidate sets used here. The details are in
[reports/04_published_comparison.md](reports/04_published_comparison.md) and I would rather state
the caveat than quote the headline.

## Results

| | DBLP-ACM (clean) | Amazon-Google (dirty) |
|---|---|---|
| Records | 2,616 x 2,294 | 1,363 x 3,226 |
| Pair space | 6,001,104 | 4,397,038 |
| True matches | 2,224 | 1,300 |
| Match rate | 0.037% | 0.030% |
| Blocking retains | 100% of matches at 0.87% of the pair space | 98.9% at 0.62% |
| Best end to end F1 | **0.990** | **0.615** |
| Best rule | tuned cut plus 1:1 constraint | tuned cut plus 1:1 constraint |
| Precision / recall at that point | 0.995 / 0.984 | 0.721 / 0.538 |

One match per 2,700 pairs on the clean dataset and one per 3,400 on the dirty one. That ratio is
the whole problem. It is why accuracy is a useless metric here, a model predicting "no match"
everywhere scores 99.97%, and it is why blocking is not an optimisation but a precondition.

## The five things worth reading

### 1. A blocking key that reads as obviously safe deletes the entire problem

Blocking on an exact match of the publication venue in DBLP-ACM returns **zero candidate pairs**.
Not few, zero. The venue is abbreviated on one side and spelled out on the other, so the two never
agree as strings. Every true match would be lost before any model saw it, and because the pipeline
would still run and still produce a confident looking output, nothing about the failure announces
itself.

The same pattern with a different cause: `manufacturer` is missing on 92.8% of the Google records,
so blocking on it retains 4.6% of matches.

| Scheme | DBLP-ACM matches kept | Amazon-Google matches kept | Share of pair space scored |
|---|---|---|---|
| exact on venue | 0.0% | not applicable | 0 pairs |
| exact on manufacturer | not applicable | 4.6% | 0.05% |
| exact on title | 91.1% | 3.9% | 0.04% / 0.003% |
| shared title token | 100% | 99.7% | 28.2% / 10.9% |
| TF-IDF character n-gram, k=1 | 97.7% | 61.0% | 0.04% / 0.03% |
| **TF-IDF character n-gram, k=20** | **100%** | **98.9%** | **0.87% / 0.62%** |
| TF-IDF character n-gram, k=50 | 100% | 99.4% | 2.18% / 1.55% |

Full table in [reports/02_blocking.md](reports/02_blocking.md).

### 2. Blocking sets a ceiling that no model can lift

Whatever share of true matches blocking discards is an upper bound on the recall of everything
downstream. On the dirty dataset that ceiling never reaches 1.0 at any sane cost: k=20 keeps
98.9%, k=50 keeps 99.4% at two and a half times the candidates, and only near cartesian blocking
reaches everything. So 1.1% of the dirty benchmark's matches are unreachable by construction, and
that is reported as part of the error budget rather than left out of the denominator.

### 3. The split experiment came out negative, and the follow up came out backwards

I expected splitting candidate pairs at random to inflate F1 relative to splitting by record,
because a random split puts pairs belonging to the same record on both sides of the divide. It
does not. Across every matcher and both datasets the gap is between -0.11 and +0.29 percentage
points, which is noise.

The explanation is that every feature here is a symmetric similarity function. Seeing one pair
involving record X during training tells the model nothing about record X, only about how similar
titles behave in general, so there is nothing to memorise.

I tested that explanation instead of asserting it, by adding one feature that does carry record
identity: a target encoding of the left record id, which is a routine thing to add and looks
harmless. The result was not the inflation I predicted. It was a collapse.

| | Similarity features only | Plus a record identity feature |
|---|---|---|
| DBLP-ACM, pair level split | 0.988 | **0.405** |
| DBLP-ACM, entity level split | 0.986 | 0.986 |
| Amazon-Google, pair level split | 0.584 | **0.207** |
| Amazon-Google, entity level split | 0.581 | 0.592 |

The mechanism, measured on one fold rather than reasoned about: the encoded feature correlates
**+0.143 with the label in training and -0.277 in test**. Because each DBLP record has exactly one
true match, a test positive's record contributes only negatives to the training fold, so its
encoding is exactly 0.0000 while training positives sit at 0.0625. The feature's meaning inverts
between fitting and prediction. Under an entity level split the same feature is constant at the
prior for unseen records, correlates exactly 0.0000 with the label, and the model ignores it.

So the useful statement is not the one I set out to demonstrate. It is that a pair level split
cannot detect an identity carrying feature at all, and the failure it hides can go in either
direction, including one that would look like a broken model with no obvious cause.

### 4. Fellegi-Sunter's weakness was its own assumption, not the method

The classical probabilistic model, which uses no labels at all, scored 0.765 F1 on DBLP-ACM at
0.619 precision when fitted on all 17 similarity features. Fitted on one feature per field, four
features, it reaches **0.901 F1 at 0.820 precision** with recall unchanged.

Four correlated views of the same title were being multiplied together as four independent pieces
of evidence, which is exactly what the conditional independence assumption forbids. Every
agreement got counted four times, so non matches that happened to share a title accumulated the
same weight as real matches.

On the dirty dataset the same fix moves F1 only from 0.448 to 0.462 and EM stops at the 200
iteration cap without converging. That is reported as it stands rather than tuned into something
tidier.

### 5. The cardinality constraint that is wrong beats the one that is right

DBLP-ACM is strictly one to one, verified rather than assumed. Amazon-Google is not: one Amazon
product matches up to five Google listings, and 144 Amazon records have more than one match.

Forcing a strict one to one assignment on the dirty dataset therefore has a measurable structural
cost. At most 1,106 of its 1,300 matches can survive it, a recall ceiling of **85.1%**. And yet:

| Decision rule | DBLP-ACM F1 | Amazon-Google F1 |
|---|---|---|
| default cut of 0.50 | 0.986 | 0.578 |
| threshold tuned on a validation split | 0.986 | 0.608 |
| tuned cut plus degree caps fitted from training labels | **0.990** | 0.611 |
| tuned cut plus strict 1:1, greedy | **0.990** | **0.615** |
| tuned cut plus strict 1:1, exact assignment | **0.990** | 0.613 |

On the dirty dataset the constraint that is factually wrong about the data wins, because the
precision it buys (0.535 to 0.721) is worth more than the recall it destroys (0.719 to 0.538)
under F1. That is a statement about F1 as much as about the data: at any operating point that
weights recall more heavily, the ranking flips. Choosing 1:1 here without saying that would be
picking a metric to justify a habit.

Also worth recording: the greedy rule and the exact assignment solver land in the same place
(0.9898 against 0.9898, 0.6147 against 0.6131). Greedy is not optimal in general, and a unit test
in this repo demonstrates a case where it is provably worse, but on this data the optimality gap
is not where the value is.

## Where the dirty benchmark actually loses matches

An F1 of 0.615 says the system is wrong without saying how. Splitting every missed match by the
stage that lost it, because the fixes are different:

| Fate of the 1,300 true matches | Count | Share |
|---|---|---|
| never proposed by blocking, unreachable by any model | 14 | 1.1% |
| proposed but scored below the cut | 365 | 28.1% |
| scored above the cut, dropped by the 1:1 constraint | 237 | 18.2% |
| correctly returned | 684 | 52.6% |

So the single largest bucket is the model, not the blocking and not the constraint. The near
misses are recognisable products written differently by the two sellers: `power director 3` against
`cyberlink power director 3`, `crystal reports xi professional edition` against
`business objects crystal reports xi`. The vendor prefix is exactly the kind of token a character
n-gram similarity is bad at discounting.

Looking at the highest scoring false positives, several do not look like errors at all:
`zonealarm internet security suite` against `zonealarm(r) internet security suite`, both priced
49.99. That is a claim about the labels rather than about the model, so I counted it instead of
asserting it: of 244 false positives, **11 (4.5%)** agree on title at a character cosine above 0.8
and carry an identical non zero price on both sides. My impression from reading the top of the
list was that this was most of them. It is not, and no number in this project assumes those 11 are
matches.

Full analysis with examples in [reports/03_error_analysis.md](reports/03_error_analysis.md).

## Next to the published figures

| | This project | Published Magellan | Published best deep model |
|---|---|---|---|
| DBLP-ACM | 0.990 | 0.984 | 0.984 |
| Amazon-Google | 0.615 | 0.491 | 0.693 |

Published figures transcribed from Table 3 of Mudgal et al., *Deep Learning for Entity Matching: A
Design Space Exploration*, SIGMOD 2018
([pdf](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)).

These are not the same measurement. Their numbers come from a supplied set of labelled candidate
pairs, 12,363 for DBLP-ACM and 11,460 for Amazon-Google, split 3:1:1. Blocking sits outside their
measurement and inside mine. Their candidate sets are also much better balanced, 18.0% and 10.2%
matches against 4.3% and 4.7% in the sets used here, and precision is easier at a higher prior.
Their Amazon-Google set contains 1,167 positives where the full ground truth has 1,300, so their
recall denominator is not the whole dataset either.

What survives all of that is the shape rather than the ranking: on the clean citation benchmark a
classical pipeline lands where both the classical and the deep published results land, and they
agree with each other there. On the dirty product benchmark the published deep model is well ahead
of the published classical baseline, and this pipeline sits between them. That is the usual
finding, that deep models earn their cost on dirty text and not on structured records.

## What did not work

- **The leakage hypothesis, as originally stated.** Covered above. The prediction was wrong and the
  experiment that tested the explanation produced the opposite sign from the one I expected.
- **Fellegi-Sunter on the dirty dataset.** EM does not converge within 200 iterations and the
  independence fix that transforms the clean dataset barely moves it. I did not chase this further.
- **The exact assignment solver.** Correct, slower, and worth 0.0 to 0.2 F1 points against greedy
  here. Kept in the repo because it answers the question, not because it earned its place.
- **Reading errors by eye.** The top of the false positive list suggested the ground truth was
  substantially incomplete. Counting said 4.5%.

## Limitations

- Two datasets. Both are English, both are two sided, and both come with complete labels, which is
  the friendly case. A single source deduplication, where every record can match every other
  record, is a different problem and is not attempted here.
- No transitive closure or clustering across more than two sources.
- The error analysis uses a single fixed threshold of 0.25 with out of fold scores, rather than the
  per fold tuned cut used in the decision table. The two differ by a few hundredths and the
  breakdown is not sensitive to it, but they are not identical.
- Degree caps are fitted from training fold labels. In a deployment with no labels at all they
  would be a domain assumption, and stating the cardinality of your sources is then a design
  decision rather than a measurement.
- No pretrained language model, so this is not evidence about what is achievable on the dirty
  benchmark, only about what these methods achieve.

## Running it

```
make setup
make demo
```

`make demo` downloads both benchmarks, profiles them, runs every blocking scheme, trains every
matcher under both splitting regimes, runs the two follow up experiments, evaluates the decision
layer, produces the error analysis and writes the comparison. It takes about two and a half
minutes and reproduces every number in this README.

```
make test          # 17 tests, no network and no downloaded data required
make determinism   # runs the pipeline twice and checks the artifacts are byte identical
```

## Layout

```
config/       dataset definitions, and the published figures with their citation
src/er/       fetch, load, profile, text, blocking, features, matchers, resolve
scripts/      one entry point per phase
reports/      generated markdown, one per phase
artifacts/    generated csv, the numbers behind the reports
tests/        unit tests against in memory fixtures
```

Data is not committed. `make data` fetches both archives and records the retrieval date and the
SHA-256 of each into `data/raw/provenance.json`, so a future run that gets different content will
show it rather than silently absorb it.

## Data

Both benchmarks come from the database group at the University of Leipzig, who publish them with
complete match mappings. DBLP-ACM is a bibliographic pair, 2,616 DBLP records against 2,294 ACM
records with 2,224 known matches. Amazon-GoogleProducts is a software product pair, 1,363 against
3,226 with 1,300 known matches.

Ground truth was reconciled against the sources before anything was built: no label points at a
record id that does not exist, and no source id appears twice, so the labels are usable as they
stand. Details and field level profiling in
[reports/01_data_profile.md](reports/01_data_profile.md).
