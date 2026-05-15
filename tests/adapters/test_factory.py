import pytest

from retrocast._warnings import RetroCastFutureWarning
from retrocast.adapters import ADAPTER_MAP, ADAPTER_TYPES, adapt_single_route, get_adapter
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.exceptions import RetroCastException
from retrocast.models.chem import TargetInput


def test_get_adapter_known_adapter():
    """
    tests that a known adapter name returns an instance of baseadapter.
    we test one specific case ('aizynth') to ensure the mechanism works.
    """
    adapter_name = "aizynth"
    adapter = get_adapter(adapter_name)
    assert isinstance(adapter, BaseAdapter)
    assert isinstance(adapter, ADAPTER_TYPES[adapter_name])
    assert adapter is not ADAPTER_MAP[adapter_name]


def test_get_adapter_unknown_adapter_raises_exception():
    """
    tests that requesting an unknown adapter name raises an RetroCastException
    with a helpful error message.
    """
    unknown_name = "this-adapter-does-not-exist"
    with pytest.raises(RetroCastException, match=f"unknown adapter '{unknown_name}'") as exc_info:
        get_adapter(unknown_name)
    assert exc_info.value.code == "adapter.unknown"
    assert exc_info.value.context["adapter"] == unknown_name


def test_adapt_single_route_uses_target_local_payload_contract():
    target = TargetInput(id="target_1", smiles="CC")
    raw_routes = [
        {
            "type": "mol",
            "smiles": "CC",
            "children": [
                {
                    "type": "reaction",
                    "smiles": "",
                    "children": [{"type": "mol", "smiles": "C", "children": [], "in_stock": True}],
                    "metadata": {},
                }
            ],
            "in_stock": False,
        }
    ]

    with pytest.warns(RetroCastFutureWarning, match="adapt_single_route"):
        route = adapt_single_route(raw_routes, target, "aizynth")

    assert route is not None
    assert route.target.smiles == "CC"


@pytest.mark.parametrize("adapter_name, adapter_type", ADAPTER_TYPES.items())
def test_all_adapters_in_map_are_valid(adapter_name, adapter_type):
    """
    iterates through the entire ADAPTER_MAP to verify that each registered
    adapter can be retrieved by its key and is a valid baseadapter instance.
    this acts as a regression test for the ADAPTER_MAP constant itself.
    """
    retrieved_adapter = get_adapter(adapter_name)
    assert isinstance(retrieved_adapter, adapter_type)
    assert isinstance(retrieved_adapter, BaseAdapter)
    assert isinstance(ADAPTER_MAP[adapter_name], adapter_type)
