from collections.abc import Collection, Mapping, Sequence
from os import PathLike
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

_StrPath: TypeAlias = str | PathLike[str]
_JSONScalar: TypeAlias = None | bool | int | float | str
_JSONValue: TypeAlias = _JSONScalar | list["_JSONValue"] | dict[str, "_JSONValue"]

__version__: str
__engine__: Literal["rust"]
__all__: list[str]

class _TargetInput(TypedDict):
    id: str
    smiles: str
    inchikey: str
    acceptable_routes: NotRequired[list[dict[str, Any]]]
    annotations: NotRequired[dict[str, _JSONValue]]

class _Target(TypedDict):
    id: str
    smiles: str
    inchikey: str
    acceptable_routes: list[dict[str, Any]]
    annotations: dict[str, _JSONValue]

class _TaskInput(TypedDict):
    name: str
    targets: dict[str, _TargetInput]
    description: NotRequired[str]
    default_constraints: NotRequired[list[dict[str, _JSONValue]]]
    constraints: NotRequired[dict[str, list[dict[str, _JSONValue]]]]
    metric_label: NotRequired[str | None]
    annotations: NotRequired[dict[str, _JSONValue]]
    schema_version: NotRequired[Literal["2"]]

class _Task(TypedDict):
    name: str
    targets: dict[str, _Target]
    description: str
    default_constraints: list[dict[str, _JSONValue]]
    constraints: dict[str, list[dict[str, _JSONValue]]]
    metric_label: str | None
    annotations: dict[str, _JSONValue]
    schema_version: Literal["2"]

class _ExecutionStatsInput(TypedDict):
    wall_time: NotRequired[dict[str, float]]
    cpu_time: NotRequired[dict[str, float]]

class _ExecutionStats(TypedDict):
    wall_time: dict[str, float]
    cpu_time: dict[str, float]

class _ManifestOutput(TypedDict):
    path: _StrPath
    content_type: Literal["benchmark", "predictions", "route_corpus", "stock", "unknown"]
    value: NotRequired[_JSONValue]
    label: NotRequired[str]
    content_hash: NotRequired[str]

class NativePredictions:
    def json(self) -> str: ...
    def write(self, path: _StrPath) -> None: ...
    def to_dict(self) -> dict[str, list[dict[str, Any]]]: ...

class NativeEvaluation:
    def json(self) -> str: ...
    def write(self, path: _StrPath) -> None: ...
    def metric_label(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

def adapt(
    raw: _JSONValue,
    adapter: str,
    *,
    mode: Literal["strict", "lenient"] = "strict",
    target: _TargetInput | Mapping[str, _JSONValue] | None = None,
    source_key: str | None = None,
    max_candidates: int | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]: ...
def ingest(
    raw: _JSONValue,
    adapter: str,
    task: _TaskInput | Mapping[str, _JSONValue],
    *,
    mode: Literal["strict", "lenient"] = "strict",
    max_candidates: int | None = None,
    workers: int = 1,
) -> NativePredictions: ...
def ingest_file(
    raw_path: _StrPath,
    adapter: str,
    task_path: _StrPath,
    *,
    mode: Literal["strict", "lenient"] = "strict",
    max_candidates: int | None = None,
    workers: int = 1,
) -> NativePredictions: ...
def score(
    predictions: NativePredictions,
    task: _TaskInput | Mapping[str, _JSONValue],
    stocks: Mapping[str, Collection[str]],
    *,
    match_level: Literal["full", "no_stereo", "connectivity"] = "full",
    acceptable_route_match: Literal["prefix", "exact"] = "prefix",
    execution_stats: _ExecutionStatsInput | Mapping[str, Mapping[str, float]] | None = None,
    workers: int = 1,
) -> NativeEvaluation: ...
def analyze(
    evaluation: NativeEvaluation,
    *,
    ks: Sequence[int] = ...,
    prefix_depths: Sequence[int] = ...,
    n_boot: int = 10_000,
    seed: int = 42,
    workers: int = 1,
) -> dict[str, Any]: ...
def analyze_file(
    evaluation_path: _StrPath,
    *,
    ks: Sequence[int] = ...,
    prefix_depths: Sequence[int] = ...,
    execution_stats_path: _StrPath | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
    workers: int = 1,
) -> dict[str, Any]: ...
def evaluate(
    raw_path: _StrPath,
    benchmark_path: _StrPath,
    stock_path: _StrPath,
    output_dir: _StrPath,
    *,
    stock_name: str | None = None,
    execution_stats_path: _StrPath | None = None,
    adapter: str = "aizynthfinder",
    workers: int = 1,
    mode: Literal["strict", "lenient"] = "strict",
    max_candidates: int | None = None,
    match_level: Literal["full", "no_stereo", "connectivity"] = "full",
    acceptable_route_match: Literal["prefix", "exact"] = "prefix",
    ks: Sequence[int] = ...,
    prefix_depths: Sequence[int] = ...,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]: ...
def validate_task(value: _TaskInput | Mapping[str, _JSONValue], *, chemistry: bool = True) -> _Task: ...
def load_task(path: _StrPath, *, chemistry: bool = False) -> _Task: ...
def write_task(value: _TaskInput | Mapping[str, _JSONValue], path: _StrPath) -> None: ...
def resolve_stock_bindings(
    value: _TaskInput | Mapping[str, _JSONValue],
) -> dict[str, str | None]: ...
def load_stock(path: _StrPath, *, representation: Literal["smiles", "inchikey"] = "smiles") -> list[str]: ...
def read_json(path: _StrPath) -> _JSONValue: ...
def write_json(value: _JSONValue, path: _StrPath) -> None: ...
def write_json_gz(value: _JSONValue, path: _StrPath) -> None: ...
def validate_execution_stats(
    value: _ExecutionStatsInput | Mapping[str, Mapping[str, float]],
) -> _ExecutionStats: ...
def write_execution_stats(value: _ExecutionStatsInput | Mapping[str, Mapping[str, float]], path: _StrPath) -> None: ...
def create_manifest(
    action: str,
    sources: Sequence[_StrPath],
    outputs: Sequence[_ManifestOutput],
    root_dir: _StrPath,
    *,
    parameters: Mapping[str, _JSONValue] | None = None,
    statistics: Mapping[str, _JSONValue] | None = None,
    directives: Mapping[str, _JSONValue] | None = None,
    summary: Mapping[str, _JSONValue] | None = None,
    release_name: str | None = None,
    keyed_output_files: bool = False,
) -> dict[str, Any]: ...
def create_planner_manifest(
    action: str,
    adapter: str,
    raw_results_path: _StrPath,
    sources: Sequence[_StrPath],
    root_dir: _StrPath,
    *,
    parameters: Mapping[str, _JSONValue] | None = None,
    statistics: Mapping[str, _JSONValue] | None = None,
    summary: Mapping[str, _JSONValue] | None = None,
    release_name: str | None = None,
) -> dict[str, Any]: ...
def verify_manifest(
    manifest_path: _StrPath,
    root_dir: _StrPath,
    *,
    deep: bool = False,
    output_only: bool = False,
    lenient: bool = False,
) -> dict[str, Any]: ...
def verify_planner_manifest(
    manifest_path: _StrPath,
    root_dir: _StrPath,
    *,
    deep: bool = False,
    output_only: bool = False,
) -> dict[str, Any]: ...
def canonicalize_smiles(smiles: str, remove_mapping: bool = False, ignore_stereo: bool = False) -> str: ...
def get_inchi_key(smiles: str, level: Literal["full", "no_stereo", "connectivity"] = "full") -> str: ...
def reduce_inchi_key(inchikey: str, level: Literal["full", "no_stereo", "connectivity"]) -> str: ...
def molecular_descriptors(smiles: str) -> tuple[int, float, int]: ...
def engine_info() -> tuple[str, Literal["RDKit C++"], str]: ...
