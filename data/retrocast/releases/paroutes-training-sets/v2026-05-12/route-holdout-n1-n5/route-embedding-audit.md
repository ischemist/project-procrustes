# route embedding audit: route-holdout-n1-n5

- match level: `full`
- allow leaf extension: `true`
- partial minimum reactions: 2 reactions
- training routes: 169 126
- training route signatures: 163 636
- training reaction signatures: 306 049

terms: a query route is the route being searched for. a container route is a training route where an embedding is found.

| metric | mkt-cnv-160 | n1-routes | n5-routes |
| --- | ---: | ---: | ---: |
| query routes | 160 | 10 000 | 10 000 |
| reaction signatures in training | 380 / 733 (51.8%) | 8 879 / 30 877 (28.8%) | 11 613 / 35 893 (32.4%) |
| exact route signatures | 0 / 160 (0.0%) | 0 / 10 000 (0.0%) | 0 / 10 000 (0.0%) |
| full route embeddings | 3 / 160 (1.9%) | 276 / 10 000 (2.8%) | 218 / 10 000 (2.2%) |
| routes with internal subroute embedding | 78 / 160 (48.8%) | 2 508 / 10 000 (25.1%) | 3 765 / 10 000 (37.6%) |

## mkt-cnv-160

### overlap summary

380 / 733 (51.8%) reaction signatures are present in training; 0 / 160 (0.0%) query routes have exact route-signature matches.

3 / 160 (1.9%) query routes are fully embedded somewhere inside training routes. these produce 3 total matching occurrences. 3 query routes have a root-shifted full embedding; 1 query route has a leaf-extended full embedding. 0 occurrences share the training target (distance 0 from the root); 1 occurrence is embedded at distance 1 (a prefix of a root child); 2 occurrences are embedded at distance 2 (a prefix of a child of a root child).

78 / 160 (48.8%) query routes have at least one embedded internal subroute of 2+ reactions. there are 274 non-root internal subroutes with at least 2 reactions. 154 / 274 (56.2%) of those internal subroutes are embedded. there are 1 825 total partial matching occurrences (11.85 matches per embedded internal subroute).

### best-match coverage

when overlap exists, how large and how specific is the largest overlap? each query route is compared to the single container route that embeds the largest part of it.

basis: full routes + internal subroutes with 2+ reactions.

#### matched fraction of query route

matched reactions divided by all reactions in the query route.

| population | mean | median | p90 |
| --- | ---: | ---: | ---: |
| all query routes (160) | 26.7% | 0.0% | 66.7% |
| query routes with any embedding (78) | 54.8% | 50.0% | 83.3% |

#### matched fraction of training route

matched reactions divided by all reactions in the container route.

| population | mean | median | p90 |
| --- | ---: | ---: | ---: |
| all query routes (160) | 37.8% | 0.0% | 100.0% |
| query routes with any embedding (78) | 77.6% | 75.0% | 100.0% |

query routes with any embedding match 13.60 unique training routes and 23.44 total occurrences per query route on average.

### largest embedded fraction of query-route reactions

largest embedded match reactions divided by all reactions in the query route.

| largest embedded fraction of query-route reactions | query routes |
| --- | ---: |
| 0% | 82 |
| (0,25%] | 0 |
| (25,50%] | 44 |
| (50,75%] | 22 |
| (75,100%) | 9 |
| 100% | 3 |

## n1-routes

### overlap summary

8 879 / 30 877 (28.8%) reaction signatures are present in training; 0 / 10 000 (0.0%) query routes have exact route-signature matches.

276 / 10 000 (2.8%) query routes are fully embedded somewhere inside training routes. these produce 541 total matching occurrences. 181 query routes have a root-shifted full embedding; 105 query routes have a leaf-extended full embedding. 100 occurrences share the training target (distance 0 from the root); 299 occurrences are embedded at distance 1 (a prefix of a root child); 98 occurrences are embedded at distance 2 (a prefix of a child of a root child); 27 occurrences are embedded at distance 3 (a prefix of a depth-3 subtree); 10 occurrences are embedded at distance 4 (a prefix of a depth-4 subtree); 5 occurrences are embedded at distance 5 (a prefix of a depth-5 subtree); 2 occurrences are embedded at distance 6 (a prefix of a depth-6 subtree).

2 508 / 10 000 (25.1%) query routes have at least one embedded internal subroute of 2+ reactions. there are 11 216 non-root internal subroutes with at least 2 reactions. 3 933 / 11 216 (35.1%) of those internal subroutes are embedded. there are 25 709 total partial matching occurrences (6.54 matches per embedded internal subroute).

### best-match coverage

when overlap exists, how large and how specific is the largest overlap? each query route is compared to the single container route that embeds the largest part of it.

basis: full routes + internal subroutes with 2+ reactions.

#### matched fraction of query route

matched reactions divided by all reactions in the query route.

| population | mean | median | p90 |
| --- | ---: | ---: | ---: |
| all query routes (10 000) | 17.7% | 0.0% | 66.7% |
| query routes with any embedding (2 581) | 68.7% | 66.7% | 100.0% |

#### matched fraction of training route

matched reactions divided by all reactions in the container route.

| population | mean | median | p90 |
| --- | ---: | ---: | ---: |
| all query routes (10 000) | 19.1% | 0.0% | 75.0% |
| query routes with any embedding (2 581) | 74.1% | 66.7% | 100.0% |

query routes with any embedding match 6.36 unique training routes and 10.17 total occurrences per query route on average.

### largest embedded fraction of query-route reactions

largest embedded match reactions divided by all reactions in the query route.

| largest embedded fraction of query-route reactions | query routes |
| --- | ---: |
| 0% | 7 419 |
| (0,25%] | 3 |
| (25,50%] | 490 |
| (50,75%] | 1 588 |
| (75,100%) | 224 |
| 100% | 276 |

## n5-routes

### overlap summary

11 613 / 35 893 (32.4%) reaction signatures are present in training; 0 / 10 000 (0.0%) query routes have exact route-signature matches.

218 / 10 000 (2.2%) query routes are fully embedded somewhere inside training routes. these produce 285 total matching occurrences. 138 query routes have a root-shifted full embedding; 85 query routes have a leaf-extended full embedding. 87 occurrences share the training target (distance 0 from the root); 131 occurrences are embedded at distance 1 (a prefix of a root child); 49 occurrences are embedded at distance 2 (a prefix of a child of a root child); 11 occurrences are embedded at distance 3 (a prefix of a depth-3 subtree); 6 occurrences are embedded at distance 4 (a prefix of a depth-4 subtree); 1 occurrence is embedded at distance 5 (a prefix of a depth-5 subtree).

3 765 / 10 000 (37.6%) query routes have at least one embedded internal subroute of 2+ reactions. there are 17 377 non-root internal subroutes with at least 2 reactions. 6 688 / 17 377 (38.5%) of those internal subroutes are embedded. there are 47 579 total partial matching occurrences (7.11 matches per embedded internal subroute).

### best-match coverage

when overlap exists, how large and how specific is the largest overlap? each query route is compared to the single container route that embeds the largest part of it.

basis: full routes + internal subroutes with 2+ reactions.

#### matched fraction of query route

matched reactions divided by all reactions in the query route.

| population | mean | median | p90 |
| --- | ---: | ---: | ---: |
| all query routes (10 000) | 25.0% | 0.0% | 75.0% |
| query routes with any embedding (3 781) | 66.2% | 66.7% | 83.3% |

#### matched fraction of training route

matched reactions divided by all reactions in the container route.

| population | mean | median | p90 |
| --- | ---: | ---: | ---: |
| all query routes (10 000) | 27.9% | 0.0% | 80.0% |
| query routes with any embedding (3 781) | 73.8% | 75.0% | 100.0% |

query routes with any embedding match 7.55 unique training routes and 12.66 total occurrences per query route on average.

### largest embedded fraction of query-route reactions

largest embedded match reactions divided by all reactions in the query route.

| largest embedded fraction of query-route reactions | query routes |
| --- | ---: |
| 0% | 6 219 |
| (0,25%] | 5 |
| (25,50%] | 1 019 |
| (50,75%] | 2 050 |
| (75,100%) | 489 |
| 100% | 218 |
