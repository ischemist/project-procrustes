from __future__ import annotations

from pathlib import Path

import pytest

from retrocast.adapters.ursa_llm_adapter import UrsaAdapter
from retrocast.chem import canonicalize_smiles
from retrocast.io.blob import load_json_artifact
from retrocast.models.benchmark import create_benchmark, create_benchmark_target
from retrocast.workflow.ingest import ingest_model_predictions

RAW_TARGET_SMILES = {
    "Ebastine": "CC(C)(C)C1=CC=C(C=C1)C(=O)CCCN2CCC(CC2)OC(C3=CC=CC=C3)C4=CC=CC=C4",
    "Sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    "Tivozanib": "COC1=C(OC)C=C2C(OC3=CC(Cl)=C(NC(=O)NC4=NOC(C)=C4)C=C3)=CC=NC2=C1",
}
RAW_RESULTS_PATH = Path(__file__).resolve().parents[1] / "testing_data/model-predictions/ursa-llm/completions.jsonl"


@pytest.mark.integration
def test_ingest_ursa_llm_matches_benchmark_targets_by_canonical_smiles(tmp_path):
    benchmark = create_benchmark(
        name="ursa-llm-fixture",
        stock=set(),
        targets={
            target_id: create_benchmark_target(id=target_id, smiles=raw_smiles)
            for target_id, raw_smiles in RAW_TARGET_SMILES.items()
        },
    )
    raw_ursa_llm_data = load_json_artifact(RAW_RESULTS_PATH)

    processed_routes, save_path, stats = ingest_model_predictions(
        model_name="ursa-llm",
        benchmark=benchmark,
        raw_data=raw_ursa_llm_data,
        adapter=UrsaAdapter(),
        output_dir=tmp_path,
        provider_output_kind="provider_output",
    )

    assert save_path == Path(tmp_path) / benchmark.name / "ursa-llm" / "routes.json.gz"
    assert stats.targets_with_at_least_one_route == set(RAW_TARGET_SMILES)
    assert stats.successful_routes_before_dedup >= len(RAW_TARGET_SMILES)

    for target_id, raw_smiles in RAW_TARGET_SMILES.items():
        expected_smiles = canonicalize_smiles(raw_smiles)
        routes = processed_routes[target_id]
        assert routes
        assert all(route.target.smiles == expected_smiles for route in routes)
