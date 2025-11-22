
Create USPTO-190 in Benchmark format:

```bash
retrocast create-benchmark --input data/uspto-190.csv --output data/1-benchmarks/definitions/uspto-190 --name "USPTO 190" --stock-name "buyables-stock"
```

Verify integrity:

```bash
retrocast verify --target data/1-benchmarks/definitions/uspto-190.manifest.json
```

### Casting PaRoutes

```bash
uv run scripts/paroutes/01-cast-paroutes.py
uv run scripts/paroutes/01-cast-paroutes.py --check-buyables
```

```bash
retrocast verify --target data/1-benchmarks/definitions/paroutes-n1-full.manifest.json
retrocast verify --target data/1-benchmarks/definitions/paroutes-n5-full.manifest.json

retrocast verify --target data/1-benchmarks/definitions/paroutes-n1-full-buyables.manifest.json
retrocast verify --target data/1-benchmarks/definitions/paroutes-n5-full-buyables.manifest.json
```

### Testing seed stability

```bash
uv run scripts/paroutes/02-create-subsets.py
bash scripts/directmultistep/run-ingest.sh
retrocast score --all-models --all-datasets
```

Create plots:
```bash
uv run scripts/06-check-seed-stability.py --model dms-explorer-xl --base-benchmark ref-lin-600 --seeds 42 299792458 19910806 20260317 17760704 17890304 20251030 662607015 20180329 20170612 20180818 20151225 19690721 20160310 19450716 --stock n5-stock

uv run scripts/06-check-seed-stability.py --model dms-explorer-xl --base-benchmark ref-cnv-400 --seeds 42 299792458 19910806 20260317 17760704 17890304 20251030 662607015 20180329 20170612 20180818 20151225 19690721 20160310 19450716 --stock n5-stock

uv run scripts/06-check-seed-stability.py --model dms-explorer-xl --base-benchmark mkt-lin-500 --seeds 42 299792458 19910806 20260317 17760704 17890304 20251030 662607015 20180329 20170612 20180818 20151225 19690721 20160310 19450716 --stock buyables-stock

uv run scripts/06-check-seed-stability.py --model dms-explorer-xl --base-benchmark mkt-cnv-160 --seeds 42 299792458 19910806 20260317 17760704 17890304 20251030 662607015 20180329 20170612 20180818 20151225 19690721 20160310 19450716 --stock buyables-stock
```

ref-lin-600 Stability Statistics
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric      ┃ Mean (%) ┃ Std Dev ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ Solvability │    74.17 │   1.292 │
│ Top-1       │    28.53 │   1.723 │
│ Top-10      │    41.81 │   1.538 │
└─────────────┴──────────┴─────────┘
 Seed Representativeness (Lowest Deviation is Best)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃   Seed    ┃ Deviation Score ┃ Z-Scores (Top1, Solv, Top10) ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    1 │ 17890304  │          0.5915 │           (+0.5, +0.3, +0.6) │
│    2 │    42     │          0.6197 │           (+0.5, +0.0, -0.6) │
│    3 │ 17760704  │          1.2902 │           (+0.7, -0.6, +0.7) │
│    4 │ 299792458 │          1.3693 │           (-0.1, -1.2, -0.1) │
│    5 │ 20151225  │          1.8066 │           (-1.3, +0.0, -0.4) │
└──────┴───────────┴─────────────────┴──────────────────────────────┘

ref-cnv-400 Stability Statistics
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric      ┃ Mean (%) ┃ Std Dev ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ Solvability │    74.05 │   1.385 │
│ Top-1       │    33.85 │   0.757 │
│ Top-10      │    45.32 │   1.374 │
└─────────────┴──────────┴─────────┘
         Seed Representativeness (Lowest Deviation is Best)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃   Seed    ┃ Deviation Score ┃ Z-Scores (Top1, Solv, Top10) ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    1 │ 662607015 │          0.1393 │           (+0.2, -0.2, -0.2) │
│    2 │ 19450716  │          0.2752 │           (-0.1, +0.5, -0.0) │
│    3 │    42     │          1.1998 │           (+0.9, -0.0, +0.7) │
│    4 │ 20151225  │          1.2893 │           (-0.5, -0.6, +0.9) │
│    5 │ 20260317  │          1.3805 │           (-1.1, +0.1, +0.3) │
└──────┴───────────┴─────────────────┴──────────────────────────────┘

mkt-lin-500 Stability Statistics
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric      ┃ Mean (%) ┃ Std Dev ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ Solvability │    87.79 │   1.367 │
│ Top-1       │    31.83 │   1.775 │
│ Top-10      │    49.92 │   1.726 │
└─────────────┴──────────┴─────────┘
         Seed Representativeness (Lowest Deviation is Best)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃   Seed    ┃ Deviation Score ┃ Z-Scores (Top1, Solv, Top10) ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    1 │ 19450716  │          0.8285 │           (+0.5, +0.6, -0.4) │
│    2 │ 20251030  │          1.0516 │           (+0.5, +0.4, +0.7) │
│    3 │ 20160310  │          1.1889 │           (-0.2, -0.4, +1.0) │
│    4 │ 662607015 │          1.2414 │           (+0.5, +0.7, +0.6) │
│    5 │ 19910806  │          1.4432 │           (-0.1, +1.2, -0.2) │
└──────┴───────────┴─────────────────┴──────────────────────────────┘

mkt-cnv-160 Stability Statistics
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric      ┃ Mean (%) ┃ Std Dev ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ Solvability │    86.29 │   1.715 │
│ Top-1       │    32.54 │   1.832 │
│ Top-10      │    50.96 │   1.766 │
└─────────────┴──────────┴─────────┘
         Seed Representativeness (Lowest Deviation is Best)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank ┃   Seed    ┃ Deviation Score ┃ Z-Scores (Top1, Solv, Top10) ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    1 │ 20180329  │          0.0284 │           (-0.0, -0.0, +0.2) │
│    2 │ 19690721  │          0.1435 │           (-0.0, +0.3, +0.2) │
│    3 │ 662607015 │          0.4108 │           (-0.0, +0.3, -0.5) │
│    4 │ 20151225  │          0.5326 │           (-0.0, +0.7, -0.2) │
│    5 │ 20260317  │          1.2371 │           (-1.0, +0.3, +0.2) │
└──────┴───────────┴─────────────────┴──────────────────────────────┘
