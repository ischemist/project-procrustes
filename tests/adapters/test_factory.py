import pytest

from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.factory import ADAPTER_MAP, get_adapter
from retrocast.exceptions import RetroCastException


def test_get_adapter_known_adapter():
    """
    tests that a known adapter name returns an instance of baseadapter.
    we test one specific case ('aizynth') to ensure the mechanism works.
    """
    adapter_name = "aizynth"
    adapter = get_adapter(adapter_name)
    assert isinstance(adapter, BaseAdapter)
    # check if it's the correct specific class instance from the map
    assert isinstance(adapter, type(ADAPTER_MAP[adapter_name]))


def test_get_adapter_unknown_adapter_raises_exception():
    """
    tests that requesting an unknown adapter name raises an RetroCastException
    with a helpful error message.
    """
    unknown_name = "this-adapter-does-not-exist"
    with pytest.raises(RetroCastException, match=f"unknown adapter '{unknown_name}'"):
        get_adapter(unknown_name)


@pytest.mark.parametrize("adapter_name, adapter_instance", ADAPTER_MAP.items())
def test_all_adapters_in_map_are_valid(adapter_name, adapter_instance):
    """
    iterates through the entire ADAPTER_MAP to verify that each registered
    adapter can be retrieved by its key and is a valid baseadapter instance.
    this acts as a regression test for the ADAPTER_MAP constant itself.
    """
    retrieved_adapter = get_adapter(adapter_name)
    assert retrieved_adapter is adapter_instance
    assert isinstance(retrieved_adapter, BaseAdapter)
