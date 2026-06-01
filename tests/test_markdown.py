from __future__ import annotations

import pytest

from retrocast.markdown import markdown_table


def test_markdown_table_formats_alignment_and_cells() -> None:
    assert markdown_table(
        ["name", "count", "rate"],
        [("route", 12, "50.0%")],
        align=["left", "right", "center"],
    ) == "\n".join(
        [
            "| name | count | rate |",
            "| --- | ---: | :---: |",
            "| route | 12 | 50.0% |",
        ]
    )


def test_markdown_table_rejects_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="alignment"):
        markdown_table(["name"], [], align=["left", "right"])

    with pytest.raises(ValueError, match="rows"):
        markdown_table(["name"], [("route", "extra")])
