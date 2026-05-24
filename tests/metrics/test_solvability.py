import pytest

from retrocast.exceptions import InputError, UnsupportedValidityTierError
from retrocast.metrics.solvability import get_route_tier_failure_codes, is_route_tier_valid
from tests.helpers import _make_simple_route


@pytest.mark.unit
class TestTierValidity:
    def test_tier_zero_uses_generic_route_api(self):
        route = _make_simple_route("CC", "C")

        assert get_route_tier_failure_codes(route, 0) == []
        assert is_route_tier_valid(route, 0) is True

    def test_unimplemented_tier_raises_typed_error(self):
        route = _make_simple_route("CC", "C")

        with pytest.raises(UnsupportedValidityTierError) as exc_info:
            get_route_tier_failure_codes(route, 1)

        assert exc_info.value.code == "validity.unsupported_tier"
        assert exc_info.value.context == {"tier": 1, "implemented_tiers": [0]}

    def test_unknown_tier_raises_typed_error(self):
        route = _make_simple_route("CC", "C")

        with pytest.raises(InputError) as exc_info:
            get_route_tier_failure_codes(route, 99)

        assert exc_info.value.code == "validity.unknown_tier"
        assert exc_info.value.context == {"tier": 99, "supported_tiers": [0, 1, 2, 3]}
