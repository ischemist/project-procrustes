
Create USPTO-190 in Benchmark format:

```bash
retrocast create-benchmark --input data/uspto-190.csv --output data/1-benchmarks/definitions/uspto-190 --name "USPTO 190"
```

Verify integrity:

```bash
retrocast verify --target data/1-benchmarks/definitions/uspto-190.manifest.json
```
