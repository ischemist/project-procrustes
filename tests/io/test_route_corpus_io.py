import pytest

from retrocast.exceptions import RetroCastIOError
from retrocast.io.blob import save_jsonl_gz
from retrocast.io.data import iter_route_corpus, load_route_corpus, save_route_corpus
from retrocast.models.chem import PredictedRoute
from tests.helpers import _make_simple_route


@pytest.mark.unit
class TestRouteCorpusIO:
    def test_route_corpus_roundtrip(self, tmp_path):
        routes = [
            _make_simple_route("CC", "C"),
            _make_simple_route("CCC", "CC"),
        ]
        path = tmp_path / "route-corpus.jsonl.gz"

        n_rows = save_route_corpus(routes, path)
        loaded = load_route_corpus(path)
        streamed = list(iter_route_corpus(path))

        assert n_rows == 2
        assert [prediction.route for prediction in loaded] == routes
        assert [prediction.route for prediction in streamed] == routes

    def test_predicted_route_corpus_roundtrip_preserves_envelope(self, tmp_path):
        predictions = [
            PredictedRoute.from_route(_make_simple_route("CC", "C"), rank=1, metadata={"source_key": "target_1"}),
            PredictedRoute.from_route(_make_simple_route("CCC", "CC"), rank=2, metadata={"source_key": "target_2"}),
        ]
        path = tmp_path / "predicted-route-corpus.jsonl.gz"

        n_rows = save_route_corpus(predictions, path)
        loaded = load_route_corpus(path)

        assert n_rows == 2
        assert loaded == predictions

    def test_invalid_route_corpus_shape_raises_format_error(self, tmp_path):
        path = tmp_path / "route-corpus.jsonl.gz"
        save_jsonl_gz([{"not": "a route"}], path)

        with pytest.raises(RetroCastIOError) as exc_info:
            load_route_corpus(path)

        assert exc_info.value.code == "io.invalid_artifact_shape"

    def test_legacy_route_rank_field_is_ignored(self, tmp_path):
        path = tmp_path / "legacy-route-corpus.jsonl.gz"
        legacy_route = _make_simple_route("CC", "C").model_dump(mode="json")
        legacy_route["rank"] = 7
        save_jsonl_gz([legacy_route], path)

        loaded = load_route_corpus(path)

        assert len(loaded) == 1
        assert loaded[0].rank is None
        assert loaded[0].route.target.smiles == "CC"
