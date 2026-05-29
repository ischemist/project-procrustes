import pytest
from pydantic import ValidationError

from retrocast.chem import get_inchi_key
from retrocast.typing import SmilesStr
from retrocast.v2.models.task import Benchmark, Target, Task, TaskConstraints


@pytest.mark.unit
def test_target_uses_canonical_identity_fields() -> None:
    target = Target(id="ethanol", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO"))

    assert target.id == "ethanol"
    assert target.inchikey == get_inchi_key("CCO")
    assert target.acceptable_routes == []


@pytest.mark.unit
def test_target_rejects_mismatched_identity_fields() -> None:
    with pytest.raises(ValidationError):
        Target(id="bad", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCC"))


@pytest.mark.unit
def test_target_rejects_invalid_smiles() -> None:
    with pytest.raises(ValidationError):
        Target(id="bad", smiles=SmilesStr("not-smiles"), inchikey=get_inchi_key("CCO"))


@pytest.mark.unit
def test_task_rejects_other_schema_versions() -> None:
    target = Target(id="ethanol", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO"))
    task = Task(name="one-target", targets={target.id: target})

    assert task.schema_version == "2"
    assert isinstance(task.default_constraints, TaskConstraints)
    with pytest.raises(ValidationError):
        Task.model_validate({"name": "bad", "targets": {}, "schema_version": "3"})


@pytest.mark.unit
def test_task_rejects_target_key_mismatch() -> None:
    target = Target(id="ethanol", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO"))

    with pytest.raises(ValidationError):
        Task(name="bad-target-map", targets={"wrong": target})


@pytest.mark.unit
def test_task_constraints_derive_required_leaf_inchikeys_from_smiles() -> None:
    constraints = TaskConstraints(required_leaves_smiles=[SmilesStr("CCO")])

    assert constraints.required_leaf_inchikeys == (get_inchi_key("CCO"),)


@pytest.mark.unit
def test_benchmark_is_a_task_with_description() -> None:
    benchmark = Benchmark(name="bench", description="small", targets={})

    assert benchmark.name == "bench"
    assert benchmark.description == "small"
