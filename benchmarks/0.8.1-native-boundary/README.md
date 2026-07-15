# v0.8.1 native-boundary benchmark

This benchmark compares the repaired v0.8.1 pipeline with both the pre-Rust v0.7.1 implementation and the first Rust-backed v0.8.0 release. It runs the complete ingest, score, and analyze workflow on the same machine and inputs. The two fixtures exercise materially different adapters: AiZynthFinder is a small route-list input, while ASKCOS is a much larger graph-shaped input.

v0.7.1 is the measured 0.7.x baseline. The later v0.7.2 tag only changes malformed SynPlanner route handling and the Tornado dependency, neither of which participates in these AiZynthFinder or ASKCOS runs.

These are single measured release-build runs. Wall time includes process startup and artifact IO. Peak RSS is the largest resident set reported by `/usr/bin/time -lp`. Throughput is the number of candidate slots divided by end-to-end wall time.

| adapter | version and front end | workers | candidates | wall (s) | candidates/s | peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AiZynthFinder | v0.7.1 Python | — | 1,830 | 8.026 | 228.0 | 311.0 |
| AiZynthFinder | v0.8.0 Python | — | 1,830 | 5.134 | 356.5 | 464.8 |
| AiZynthFinder | v0.8.1 Python | 1 | 1,830 | 3.750 | 488.0 | 152.7 |
| AiZynthFinder | v0.8.1 Python | 12 | 1,830 | 1.780 | 1,028.1 | 158.8 |
| AiZynthFinder | v0.8.1 standalone | 1 | 1,830 | 3.730 | 490.6 | 116.1 |
| AiZynthFinder | v0.8.1 standalone | 12 | 1,830 | 1.640 | 1,115.9 | 122.4 |
| ASKCOS | v0.7.1 Python | — | 25,762 | 558.189 | 46.2 | 2,380.0 |
| ASKCOS | v0.8.0 Python | — | 25,762 | 241.815 | 106.5 | 8,765.1 |
| ASKCOS | v0.8.1 Python | 1 | 25,762 | 45.860 | 561.8 | 508.0 |
| ASKCOS | v0.8.1 Python | 12 | 25,762 | 19.360 | 1,330.7 | 554.8 |
| ASKCOS | v0.8.1 standalone | 12 | 25,762 | 19.260 | 1,337.6 | 524.5 |

## Aggregate comparison

Aggregate throughput divides the 27,592 candidate slots by summed wall time. Mean peak RSS is the unweighted arithmetic mean of the two per-fixture peaks; peaks cannot be summed because the fixtures ran separately.

| version and front end | workers | summed wall (s) | aggregate candidates/s | mean peak RSS (MiB) | speedup vs v0.7.1 | RSS vs v0.7.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.7.1 Python | — | 566.214 | 48.7 | 1,345.5 | 1.0x | 1.00x |
| v0.8.0 Python | — | 246.949 | 111.7 | 4,615.0 | 2.3x | 3.43x |
| v0.8.1 Python | 1 | 49.610 | 556.2 | 330.4 | 11.4x | 0.25x |
| v0.8.1 Python | 12 | 21.140 | 1,305.2 | 356.8 | 26.8x | 0.27x |
| v0.8.1 standalone | 12 | 20.900 | 1,320.2 | 323.4 | 27.1x | 0.24x |

The Python and standalone v0.8.1 commands are nearly identical because Python is now only an API and CLI layer over the file-native Rust pipeline. Against the pre-Rust baseline, the 12-worker Python path is 4.5x faster on AiZynthFinder and 28.8x faster on ASKCOS. It also uses 49% and 77% less peak RSS, respectively.

## Methodology boundary

v0.7.1 has no one-process `pipeline` command. Its wall value is therefore the sum of the supported `ingest`, `score`, and `analyze` commands, and its peak RSS is the maximum of those three processes. v0.8.0 was measured through the equivalent three Python stage commands. v0.8.1 uses the supported one-process `pipeline` command. This compares the real end-to-end interfaces available in each release, but it should not be mistaken for a microbenchmark of engine code alone.

All measurements were taken on the same Apple M4 Max host. Candidate artifacts agree across versions, workers, and front ends. Evaluation differences between v0.7.1/v0.8.0 and v0.8.1 are limited to serialization of default-valued fields. Analysis results have semantic parity; small differences in bootstrapped confidence intervals are expected from stochastic resampling.

Exact measurements are recorded in [`benchmark-results.json`](benchmark-results.json).
