# Where the dirty benchmark loses matches

Ground truth matches: 1,300
- never proposed by blocking: 14 (1.1 percent), unreachable by any model
- proposed but scored below the 0.25 cut: 365 (28.1 percent)
- scored above the cut but dropped by the 1:1 constraint: 237 (18.2 percent)
- correctly returned: 684 (52.6 percent)


## Highest scoring false positives

|   score | amazon                                    | google                                                                 |   amazon_price |   google_price |
|--------:|:------------------------------------------|:-----------------------------------------------------------------------|---------------:|---------------:|
|   0.949 | zonealarm internet security suite         | zonealarm(r) internet security suite                                   |          49.99 |          49.99 |
|   0.945 | mavis beacon teaches typing 16            | mavis beacon teaches typing 16 deluxe                                  |          19.99 |          34.9  |
|   0.938 | system care professional                  | system care? professional                                              |          49.95 |          49.99 |
|   0.921 | safekeeper plus                           | bling software limited safekeeper plus                                 |          39.99 |          39.99 |
|   0.918 | adobe photoshop elements 4.0 (mac)        | adobe photoshop elements 4.0 mac academic                              |          89.99 |          69.99 |
|   0.915 | instant home cooking (jewel case)         | instant home cooking                                                   |           9.99 |           9.95 |
|   0.914 | the print shop 22 sb cs by the print shop | encore software 10731 - the print shop 22 pro publisherdeluxe sb cs by |           0    |          81.97 |
|   0.906 | microsoft office & windows training       | microsoft office and windows training professional                     |          29.99 |          29.99 |

## True matches that scored just below the cut

|   score | amazon                                                                 | google                                                                 |   amazon_price |   google_price |
|--------:|:-----------------------------------------------------------------------|:-----------------------------------------------------------------------|---------------:|---------------:|
|   0.249 | chicken hunter wanted jc                                               | encore software 10760 - chicken hunter wanted - win 98 me 2000 xp      |           9.99 |           7.69 |
|   0.249 | power director 3                                                       | cyberlink power director 3                                             |          79.95 |           9.99 |
|   0.248 | perfect attorney premium                                               | perfect attorney premium (pc) cosmi                                    |          39.99 |          29.99 |
|   0.248 | watchguard serverlock manager (100 servers)                            | serverlock manager - 100 servers                                       |       14995    |       56543.9  |
|   0.246 | microsoft licenses sps extrnlconnnonemplyenglands c (h3200034)         | microsoft h32-00034 sps extrnlconnnonemplyengl&s c 805529073074        |      101516    |       55420.6  |
|   0.245 | crystal reports xi professional edition                                | business objects crystal reports xi                                    |         495    |         459.99 |
|   0.245 | power production storyboard artist 4                                   | power production power production storyboard artist software animation |           0    |         498.95 |
|   0.244 | h&r block taxcut 2006 premium federal + state with usb 256m flashdrive | h&r block taxcut premium federal and state software for windows - usb  |          39.99 |          29.95 |

## True matches blocking never proposed

| score   | amazon                                                                 | google                                                                 |   amazon_price |   google_price |
|:--------|:-----------------------------------------------------------------------|:-----------------------------------------------------------------------|---------------:|---------------:|
|         | punch 5 in 1 home design                                               | punch software 20100 - punch! home design - complete product - archite |          39.99 |          36.97 |
|         | punch! 5 in 1 home design                                              | punch software 24100 - punch! 5 in 1 home design - complete product -  |          39.99 |          35.97 |
|         | microsoft windows terminal server 2003 client additional license for u | win 2003 ter svr cal 5pk microsoft r19-00846                           |         669    |         762.95 |
|         | adobe after effects professional 6.5 upgrade from standard to professi | adobe software 22070152 after effects 6.5 pbupgrd                      |         499.99 |         507    |
|         | quickbooks premier non-profit edition 2005                             | intuit inc 284216 qckbks prem nonprofit ed 2005                        |         499.95 |         404    |
|         | adobe photoshop cs2 (mac) [old version]                                | adobe systems inc 13102124 adobe photoshop cs 2 mac os x v.10.2.8 to 1 |         649    |         788.63 |
|         | cell phone software solution                                           | datapilot universal pro kit (win 2000 xp vista/mac 10.3.8 or later)    |         113.1  |          79.95 |
|         | iplaymusic beginner guitar lessons for the mac and ipod                | wingnuts 2: raina's revenge                                            |          49.99 |          28.99 |

## Mean feature values by outcome

| outcome        |   title_cosine_char |   title_jaccard |   price_rel_diff |   manufacturer_either_missing |   description_cosine_word |
|:---------------|--------------------:|----------------:|-----------------:|------------------------------:|--------------------------:|
| false negative |               0.646 |           0.399 |            0.282 |                         0.879 |                     0.251 |
| false positive |               0.766 |           0.544 |            0.288 |                         0.955 |                     0.348 |
| true negative  |               0.316 |           0.147 |            0.588 |                         0.947 |                     0.089 |
| true positive  |               0.776 |           0.575 |            0.213 |                         0.949 |                     0.274 |

## How many false positives look like missing labels

Of 244 false positives, 11 (4.5 percent) agree on title at a character cosine of 0.8 or higher and carry an identical non zero price on both sides. These are flagged for manual review rather than counted as correct: the benchmark's precision figure is a lower bound if they are genuine matches, and no reported number in this project assumes they are.
