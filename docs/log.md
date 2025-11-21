
Create USPTO-190 in Benchmark format:

```bash
retrocast create-benchmark --input data/uspto-190.csv --output data/1-benchmarks/definitions/uspto-190 --name "USPTO 190"
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
