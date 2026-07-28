from collections.abc import Collection, Mapping, Sequence
from os import PathLike
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

StrPath: TypeAlias = str | PathLike[str]
JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

__version__: str
__engine__: Literal["rust"]
__all__: list[str]

class Target(TypedDict):
    id: str
    smiles: str
    inchikey: str
    acceptable_routes: NotRequired[list[dict[str, Any]]]
    annotations: NotRequired[dict[str, JSONValue]]

class Task(TypedDict):
    name: str
    targets: dict[str, Target]
    description: NotRequired[str]
    default_constraints: NotRequired[list[dict[str, JSONValue]]]
    constraints: NotRequired[dict[str, list[dict[str, JSONValue]]]]
    metric_label: NotRequired[str | None]
    annotations: NotRequired[dict[str, JSONValue]]
    schema_version: NotRequired[Literal["2"]]

class ExecutionStats(TypedDict):
    wall_time: NotRequired[dict[str, float]]
    cpu_time: NotRequired[dict[str, float]]

class ManifestOutput(TypedDict):
    path: StrPath
    value: JSONValue
    content_type: Literal["benchmark", "predictions", "route_corpus", "stock", "unknown"]
    label: NotRequired[str]
    content_hash: NotRequired[str]

class NativePredictions:
    def json(self) -> str: ...
    def write(self, path: StrPath) -> None: ...
    def to_dict(self) -> dict[str, list[dict[str, Any]]]: ...

class NativeEvaluation:
    def json(self) -> str: ...
    def write(self, path: StrPath) -> None: ...
    def metric_label(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...

def adapt(
    raw: JSONValue,
    adapter: str,
    *,
    mode: Literal["strict", "lenient"] = "strict",
    target: Mapping[str, JSONValue] | None = None,
    source_key: str | None = None,
    max_candidates: int | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]: ...
def ingest(
    raw: JSONValue,
    adapter: str,
    task: Task,
    *,
    mode: Literal["strict", "lenient"] = "strict",
    max_candidates: int | None = None,
    workers: int = 1,
) -> NativePredictions: ...
def ingest_file(
    raw_path: StrPath,
    adapter: str,
    task_path: StrPath,
    *,
    mode: Literal["strict", "lenient"] = "strict",
    max_candidates: int | None = None,
    workers: int = 1,
) -> NativePredictions: ...
def score(
    predictions: NativePredictions,
    task: Task,
    stocks: Mapping[str, Collection[str]],
    *,
    match_level: Literal["full", "no_stereo", "connectivity"] = "full",
    acceptable_route_match: Literal["prefix", "exact"] = "prefix",
    execution_stats: ExecutionStats | None = None,
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
    evaluation_path: StrPath,
    *,
    ks: Sequence[int] = ...,
    prefix_depths: Sequence[int] = ...,
    execution_stats_path: StrPath | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
    workers: int = 1,
) -> dict[str, Any]: ...
def evaluate(
    raw_path: StrPath,
    benchmark_path: StrPath,
    stock_path: StrPath,
    output_dir: StrPath,
    *,
    stock_name: str | None = None,
    execution_stats_path: StrPath | None = None,
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
def validate_task(value: Task | Mapping[str, JSONValue]) -> Task: ...
def load_task(path: StrPath) -> Task: ...
def load_stock(path: StrPath, *, representation: Literal["smiles", "inchikey"] = "smiles") -> list[str]: ...
def read_json(path: StrPath) -> JSONValue: ...
def write_json(value: JSONValue, path: StrPath) -> None: ...
def write_json_gz(value: JSONValue, path: StrPath) -> None: ...
def validate_execution_stats(
    value: ExecutionStats | Mapping[str, Mapping[str, float]],
) -> ExecutionStats: ...
def write_execution_stats(value: ExecutionStats | Mapping[str, Mapping[str, float]], path: StrPath) -> None: ...
def create_manifest(
    action: str,
    sources: Sequence[StrPath],
    outputs: Sequence[ManifestOutput],
    root_dir: StrPath,
    *,
    parameters: Mapping[str, JSONValue] | None = None,
    statistics: Mapping[str, JSONValue] | None = None,
    directives: Mapping[str, JSONValue] | None = None,
    summary: Mapping[str, JSONValue] | None = None,
    release_name: str | None = None,
    keyed_output_files: bool = False,
) -> dict[str, Any]: ...
def verify_manifest(
    manifest_path: StrPath,
    root_dir: StrPath,
    *,
    deep: bool = False,
    output_only: bool = False,
    lenient: bool = True,
) -> dict[str, Any]: ...
def canonicalize_smiles(smiles: str, remove_mapping: bool = False, ignore_stereo: bool = False) -> str: ...
def get_inchi_key(smiles: str, level: Literal["full", "no_stereo", "connectivity"] = "full") -> str: ...
def reduce_inchi_key(inchikey: str, level: Literal["full", "no_stereo", "connectivity"]) -> str: ...
def molecular_descriptors(smiles: str) -> tuple[int, float, int]: ...
def engine_info() -> tuple[str, Literal["RDKit C++"], str]: ...
