# Data profile and ground truth reconciliation

## DBLP-ACM (structured)

- left records: 2,616
- right records: 2,294
- pair space: 6,001,104
- labelled matches: 2,224 (0.03706%, one in 2,698 pairs)
- duplicate ids in sources: left 0, right 0
- labels pointing at an id that does not exist: left 0, right 0
- strictly one to one: True (max matches per left record 1, per right record 1; 0 left and 0 right records have more than one)

left field profile

| field   | present   |   missing |   missing_pct |   distinct |   mean_chars |   max_chars |
|:--------|:----------|----------:|--------------:|-----------:|-------------:|------------:|
| title   | True      |         0 |             0 |       2521 |         56.4 |         212 |
| authors | True      |         0 |             0 |       2316 |         47   |         328 |
| venue   | True      |         0 |             0 |          5 |         11.4 |          25 |
| year    | True      |         0 |             0 |         10 |          4   |           4 |

right field profile

| field   | present   |   missing |   missing_pct |   distinct |   mean_chars |   max_chars |
|:--------|:----------|----------:|--------------:|-----------:|-------------:|------------:|
| title   | True      |         0 |          0    |       2230 |         56.8 |         272 |
| authors | True      |        14 |          0.61 |       2008 |         46.6 |         315 |
| venue   | True      |         0 |          0    |          5 |         35.2 |          76 |
| year    | True      |         0 |          0    |         10 |          4   |           4 |

## Amazon-GoogleProducts (dirty)

- left records: 1,363
- right records: 3,226
- pair space: 4,397,038
- labelled matches: 1,300 (0.02957%, one in 3,382 pairs)
- duplicate ids in sources: left 0, right 0
- labels pointing at an id that does not exist: left 0, right 0
- strictly one to one: False (max matches per left record 5, per right record 2; 144 left and 9 right records have more than one)

left field profile

| field        | present   |   missing |   missing_pct |   distinct |   mean_chars |   max_chars |
|:-------------|:----------|----------:|--------------:|-----------:|-------------:|------------:|
| title        | True      |         0 |          0    |       1345 |         35.5 |         144 |
| description  | True      |       115 |          8.44 |       1225 |       1282.9 |       19263 |
| manufacturer | True      |         0 |          0    |        350 |         13.1 |          45 |
| price        | True      |         0 |          0    |        243 |          4.8 |           9 |

right field profile

| field        | present   |   missing |   missing_pct |   distinct |   mean_chars |   max_chars |
|:-------------|:----------|----------:|--------------:|-----------:|-------------:|------------:|
| title        | True      |         0 |          0    |       3021 |         57   |         228 |
| description  | True      |       191 |          5.92 |       2687 |        194.9 |         253 |
| manufacturer | True      |      2994 |         92.81 |         67 |          1   |          31 |
| price        | True      |         0 |          0    |       1395 |          5.2 |          12 |
