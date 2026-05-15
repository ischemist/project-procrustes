import pytest

from retrocast.exceptions import RetroCastIOError
from retrocast.io.blob import save_jsonl_gz
from retrocast.io.data import iter_route_corpus, load_route_corpus, save_route_corpus
from tests.helpers import _make_simple_route


@pytest.mark.unit
class TestRouteCorpusIO:
    def test_route_corpus_roundtrip(self, tmp_path):
        routes = [
            _make_simple_route("CC", "C", rank=1),
            _make_simple_route("CCC", "CC", rank=2),
        ]
        path = tmp_path / "route-corpus.jsonl.gz"

        n_rows = save_route_corpus(routes, path)
        loaded = load_route_corpus(path)
        streamed = list(iter_route_corpus(path))

        assert n_rows == 2
        assert loaded == routes
        assert streamed == routes

    def test_invalid_route_corpus_shape_raises_format_error(self, tmp_path):
        path = tmp_path / "route-corpus.jsonl.gz"
        save_jsonl_gz([{"not": "a route"}], path)

        with pytest.raises(RetroCastIOError) as exc_info:
            load_route_corpus(path)

        assert exc_info.value.code == "io.invalid_artifact_shape"
