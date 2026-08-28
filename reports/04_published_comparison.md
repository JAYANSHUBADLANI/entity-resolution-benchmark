# This project next to the published figures

Published source: Mudgal et al., Deep Learning for Entity Matching: A Design Space Exploration, SIGMOD 2018.
Tables used: Table 2 (datasets), Table 3 (structured results). Transcribed 2026-08-28.

| dataset               | this project, best rule                  |   this project F1 |   candidate pairs here |   positive rate here % |   published Magellan F1 |   published best DL F1 |   published pairs |   published positive rate % |
|:----------------------|:-----------------------------------------|------------------:|-----------------------:|-----------------------:|------------------------:|-----------------------:|------------------:|----------------------------:|
| DBLP-ACM              | tuned cut + degree cap (fitted per fold) |             0.99  |                  52320 |                    4.3 |                   0.984 |                  0.984 |             12363 |                        18   |
| Amazon-GoogleProducts | tuned cut + strict 1:1 (greedy)          |             0.615 |                  27260 |                    4.7 |                   0.491 |                  0.693 |             11460 |                        10.2 |

## Why these columns are not the same measurement

The published numbers are measured on a supplied set of labelled candidate pairs, not on the full pair space. Blocking is therefore outside their measurement and inside this project's. Their candidate sets are also far better balanced, which makes precision easier: 18.0 percent of their DBLP-ACM pairs are matches and 10.2 percent of their Amazon-Google pairs, against 4.3 and 4.7 percent in the candidate sets used here. Their Amazon-Google set contains 1,167 positives where the published ground truth for the dataset has 1,300, so their recall denominator is not the full ground truth either.

Published protocol: Labelled candidate pairs split 3:1:1 into train, validation and evaluation.

The reading that survives all of that: on the clean citation benchmark a classical pipeline lands in the same place as both the published classical baseline and the published deep models, which agree with each other there. On the dirty product benchmark the published deep model is well ahead of the published classical baseline, and the pipeline here sits between the two while being measured on a harder candidate set. That is consistent with the usual finding, that deep models earn their cost on dirty text and not on structured records, but it is not evidence of beating anything.
