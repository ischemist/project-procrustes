from __future__ import annotations


def depth_group_value(group: int | str) -> int:
    if isinstance(group, int):
        return group
    if group.startswith("depth "):
        group = group.removeprefix("depth ")
    return int(group)


def depth_group_label(depth: int) -> str:
    return f"depth {depth}"


def depth_group_sort_key(group: object) -> tuple[int, int | str]:
    if isinstance(group, int | str):
        try:
            return (0, depth_group_value(group))
        except ValueError:
            pass
    return (1, str(group))
