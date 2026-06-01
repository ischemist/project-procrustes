from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

MarkdownAlign = Literal["left", "right", "center"]
MarkdownRow = Sequence[object]


def markdown_table(
    headers: MarkdownRow,
    rows: Sequence[MarkdownRow],
    *,
    align: Sequence[MarkdownAlign] | None = None,
) -> str:
    column_count = len(headers)
    alignment: Sequence[MarkdownAlign] = align if align is not None else ("left",) * column_count
    if len(alignment) != column_count:
        raise ValueError("markdown table alignment must match the header count")

    lines = [
        "| " + " | ".join(str(value) for value in headers) + " |",
        "| " + " | ".join(_alignment_marker(value) for value in alignment) + " |",
    ]
    for row in rows:
        if len(row) != column_count:
            raise ValueError("markdown table rows must match the header count")
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _alignment_marker(value: MarkdownAlign) -> str:
    if value == "right":
        return "---:"
    if value == "center":
        return ":---:"
    return "---"
