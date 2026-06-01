from __future__ import annotations

import pytest

from retrocast.markdown import markdown_table


@pytest.mark.unit
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


@pytest.mark.unit
def test_markdown_table_escapes_cell_delimiters_and_line_breaks() -> None:
    assert markdown_table(["source|name", "route"], [("line\nbreak", "a|b")]) == "\n".join(
        [
            r"| source\|name | route |",
            "| --- | --- |",
            r"| line break | a\|b |",
        ]
    )


@pytest.mark.unit
def test_markdown_table_rejects_alignment_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="alignment"):
        markdown_table(["name"], [], align=["left", "right"])


@pytest.mark.unit
def test_markdown_table_rejects_row_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="rows"):
        markdown_table(["name"], [("route", "extra")])
