import json
from collections.abc import Iterable
from typing import Any, cast

from retrocast.models.candidates import Candidate
from retrocast.models.route import Route
from retrocast.models.task import Task

CollectedCandidates = dict[str, list[Candidate]]
CollectedRoutes = dict[str, list[Route]]


class NativeCollectedCandidates(dict[str, list[Candidate]]):
    """A normal candidate mapping that materializes only when Python inspects it.

    Until then, the opaque handle keeps the ingested graph in Rust so scoring
    can consume it without a JSON round trip. Any mapping access or mutation
    materializes the candidates and drops the handle before exposing mutable
    Python objects.
    """

    def __init__(self, handle: Any) -> None:
        super().__init__()
        self._native_handle: Any | None = handle

    def native_handle(self) -> Any | None:
        """Return the untouched handle for internal pipeline handoff."""
        return self._native_handle

    def materialize(self) -> CollectedCandidates:
        """Expose a plain Python mapping and end native fast-path ownership."""
        self._ensure_materialized()
        return self

    def _ensure_materialized(self) -> None:
        handle = self._native_handle
        if handle is None:
            return
        payload = json.loads(handle.json())
        self._native_handle = None
        for target_id, candidates in payload.items():
            dict.__setitem__(
                self,
                target_id,
                [Candidate.model_validate(candidate) for candidate in candidates],
            )

    def __getitem__(self, key: str) -> list[Candidate]:
        self._ensure_materialized()
        return dict.__getitem__(self, key)

    def __setitem__(self, key: str, value: list[Candidate]) -> None:
        self._ensure_materialized()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: str) -> None:
        self._ensure_materialized()
        dict.__delitem__(self, key)

    def __iter__(self):
        self._ensure_materialized()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._ensure_materialized()
        return dict.__len__(self)

    def __contains__(self, key: object) -> bool:
        self._ensure_materialized()
        return dict.__contains__(self, key)

    def __repr__(self) -> str:
        self._ensure_materialized()
        return dict.__repr__(self)

    def __eq__(self, other: object) -> bool:
        self._ensure_materialized()
        if isinstance(other, NativeCollectedCandidates):
            other._ensure_materialized()
        return dict.__eq__(self, other)

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_materialized()
        return dict.get(self, key, default)

    def keys(self):
        self._ensure_materialized()
        return dict.keys(self)

    def items(self):
        self._ensure_materialized()
        return dict.items(self)

    def values(self):
        self._ensure_materialized()
        return dict.values(self)

    def copy(self) -> CollectedCandidates:
        self._ensure_materialized()
        return dict.copy(self)

    def clear(self) -> None:
        self._ensure_materialized()
        dict.clear(self)

    def pop(self, key: str, default: Any = None) -> Any:
        self._ensure_materialized()
        return dict.pop(self, key, default)

    def popitem(self) -> tuple[str, list[Candidate]]:
        self._ensure_materialized()
        return cast(tuple[str, list[Candidate]], dict.popitem(self))

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._ensure_materialized()
        return super().setdefault(key, cast(list[Candidate], default))

    def update(self, *args: Any, **kwargs: list[Candidate]) -> None:
        self._ensure_materialized()
        dict.update(self, *args, **kwargs)


def collect_candidates(candidates: Iterable[Candidate], task: Task) -> CollectedCandidates:
    from retrocast import _native

    payload = _native.collect_candidates_json(
        json.dumps([candidate.model_dump(mode="json", exclude_none=True) for candidate in candidates]),
        task.model_dump_json(exclude_none=True),
    )
    return {
        target_id: [Candidate.model_validate(candidate) for candidate in target_candidates]
        for target_id, target_candidates in json.loads(payload).items()
    }


def collect_routes(routes: Iterable[Route], task: Task) -> CollectedRoutes:
    from retrocast import _native

    payload = _native.collect_routes_json(
        json.dumps([route.model_dump(mode="json", exclude_none=True) for route in routes]),
        task.model_dump_json(exclude_none=True),
    )
    return {
        target_id: [Route.model_validate(route) for route in target_routes]
        for target_id, target_routes in json.loads(payload).items()
    }
