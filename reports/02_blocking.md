# Blocking: what each scheme keeps and what it throws away

## DBLP-ACM (structured)

Pair space 6,001,104, true matches 2,224.

| scheme             |   candidate_pairs |   candidates_pct_of_space |   pairs_completeness |   pair_quality |   matches_lost |
|:-------------------|------------------:|--------------------------:|---------------------:|---------------:|---------------:|
| exact on venue     |                 0 |                    0      |               0      |       0        |           2224 |
| exact on authors   |              1434 |                    0.0239 |               0.2833 |       0.439331 |           1594 |
| exact on title     |              2287 |                    0.0381 |               0.9114 |       0.886314 |            197 |
| tfidf title top-1  |              2616 |                    0.0436 |               0.9766 |       0.830275 |             52 |
| tfidf title top-5  |             13080 |                    0.218  |               0.9955 |       0.169266 |             10 |
| tfidf title top-10 |             26160 |                    0.4359 |               0.9996 |       0.084977 |              1 |
| tfidf title top-20 |             52320 |                    0.8718 |               1      |       0.042508 |              0 |
| tfidf title top-50 |            130800 |                    2.1796 |               1      |       0.017003 |              0 |
| exact on year      |            601284 |                   10.0196 |               1      |       0.003699 |              0 |
| shared title token |           1692769 |                   28.2076 |               1      |       0.001314 |              0 |
| full cartesian     |           6001104 |                  100      |               1      |       0.000371 |              0 |

## Amazon-GoogleProducts (dirty)

Pair space 4,397,038, true matches 1,300.

| scheme                |   candidate_pairs |   candidates_pct_of_space |   pairs_completeness |   pair_quality |   matches_lost |
|:----------------------|------------------:|--------------------------:|---------------------:|---------------:|---------------:|
| exact on title        |               113 |                    0.0026 |               0.0392 |       0.451327 |           1249 |
| tfidf title top-1     |              1363 |                    0.031  |               0.61   |       0.581805 |            507 |
| exact on manufacturer |              2345 |                    0.0533 |               0.0462 |       0.025586 |           1240 |
| tfidf title top-5     |              6815 |                    0.155  |               0.9269 |       0.176816 |             95 |
| tfidf title top-10    |             13630 |                    0.31   |               0.9708 |       0.09259  |             38 |
| tfidf title top-20    |             27260 |                    0.62   |               0.9892 |       0.047175 |             14 |
| tfidf title top-50    |             68150 |                    1.5499 |               0.9938 |       0.018958 |              8 |
| shared title token    |            478994 |                   10.8936 |               0.9969 |       0.002706 |              4 |
| full cartesian        |           4397038 |                  100      |               1      |       0.000296 |              0 |
