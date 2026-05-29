import pytest
from pydantic import ValidationError

from retrocast.chem import get_inchi_key
from retrocast.models.task import Benchmark, Target, Task, TaskConstraints
from retrocast.typing import SmilesStr


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
def test_task_derives_metric_label_from_effective_constraints() -> None:
    t1 = Target(id="t1", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO"))
    t2 = Target(id="t2", smiles=SmilesStr("CC"), inchikey=get_inchi_key("CC"))
    task = Task(
        name="constrained",
        targets={"t1": t1, "t2": t2},
        default_constraints=TaskConstraints(stock="buyables", route_depth=3),
        constraints={"t2": TaskConstraints(stock="buyables", required_leaves_smiles=[SmilesStr("C")])},
    )

    assert task.derived_metric_label() == "buyables+leaf+depth"


@pytest.mark.unit
def test_task_metric_label_overrides_derived_metric_label() -> None:
    target = Target(id="ethanol", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO"))
    task = Task(
        name="custom",
        targets={target.id: target},
        default_constraints=TaskConstraints(stock="buyables"),
        metric_label="bidirectional",
    )

    assert task.derived_metric_label() == "bidirectional"


@pytest.mark.unit
def test_task_metric_label_marks_multiple_stocks_as_stocks() -> None:
    t1 = Target(id="t1", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO"))
    t2 = Target(id="t2", smiles=SmilesStr("CC"), inchikey=get_inchi_key("CC"))
    task = Task(
        name="mixed",
        targets={"t1": t1, "t2": t2},
        constraints={
            "t1": TaskConstraints(stock="buyables"),
            "t2": TaskConstraints(stock="enamine"),
        },
    )

    assert task.derived_metric_label() == "stocks"


@pytest.mark.unit
def test_task_metric_label_without_stock_uses_leaf_and_depth_constraints() -> None:
    t1 = Target(id="t1", smiles=SmilesStr("CCO"), inchikey=get_inchi_key("CCO"))
    t2 = Target(id="t2", smiles=SmilesStr("CC"), inchikey=get_inchi_key("CC"))
    task = Task(
        name="bidirectional",
        targets={"t1": t1, "t2": t2},
        constraints={
            "t1": TaskConstraints(required_leaves_smiles=[SmilesStr("C")]),
            "t2": TaskConstraints(route_depth=2),
        },
    )

    assert task.derived_metric_label() == "leaf+depth"


@pytest.mark.unit
def test_benchmark_is_a_task_with_description() -> None:
    benchmark = Benchmark(name="bench", description="small", targets={})

    assert benchmark.name == "bench"
    assert benchmark.description == "small"
