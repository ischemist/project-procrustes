from retrocast.visualization.report import create_analysis_table, generate_markdown_report

__all__ = [
    "RouteStats",
    "create_analysis_table",
    "create_route_comparison_figure",
    "depth_group_label",
    "depth_group_sort_key",
    "depth_group_value",
    "extract_route_stats",
    "generate_markdown_report",
    "plot_analysis_report",
]


def __getattr__(name: str):
    if name == "plot_analysis_report":
        from retrocast.visualization.plots import plot_analysis_report

        return plot_analysis_report
    if name in {"RouteStats", "create_route_comparison_figure", "extract_route_stats"}:
        from retrocast.visualization.routes import RouteStats, create_route_comparison_figure, extract_route_stats

        return {
            "RouteStats": RouteStats,
            "create_route_comparison_figure": create_route_comparison_figure,
            "extract_route_stats": extract_route_stats,
        }[name]
    if name in {"depth_group_label", "depth_group_sort_key", "depth_group_value"}:
        from retrocast.visualization.depth import depth_group_label, depth_group_sort_key, depth_group_value

        return {
            "depth_group_label": depth_group_label,
            "depth_group_sort_key": depth_group_sort_key,
            "depth_group_value": depth_group_value,
        }[name]
    raise AttributeError(name)
