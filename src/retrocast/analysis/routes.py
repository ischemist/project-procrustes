"""Route analysis and visualization functions."""

from collections import defaultdict
from dataclasses import dataclass

import plotly.graph_objects as go
from ischemist.plotly import Styler
from plotly.subplots import make_subplots

from retrocast.domain.chem import get_heavy_atom_count, get_molecular_weight
from retrocast.schemas import Route


@dataclass
class RouteStats:
    """Statistics for a single route."""

    depth: int
    target_hac: int
    target_mw: float


def extract_route_stats(routes: dict[str, list[Route]]) -> list[RouteStats]:
    """
    Extract statistics from routes.

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.

    Returns:
        List of RouteStats, one per route.
    """
    stats = []
    for route_list in routes.values():
        for route in route_list:
            smiles = route.target.smiles
            stats.append(
                RouteStats(
                    depth=route.depth,
                    target_hac=get_heavy_atom_count(smiles),
                    target_mw=get_molecular_weight(smiles),
                )
            )
    return stats


def _add_violin_traces(
    fig: go.Figure,
    n1_stats: list[RouteStats],
    n5_stats: list[RouteStats],
    all_depths: list[int],
    attr: str,
    row: int,
    n1_color: str,
    n5_color: str,
) -> None:
    """Add paired violin traces for a given attribute."""
    for depth in all_depths:
        n1_vals = [getattr(s, attr) for s in n1_stats if s.depth == depth]
        n5_vals = [getattr(s, attr) for s in n5_stats if s.depth == depth]

        for vals, name, color, side, group in [
            (n1_vals, "n1 set", n1_color, "negative", "n1"),
            (n5_vals, "n5 set", n5_color, "positive", "n5"),
        ]:
            if vals:
                fig.add_trace(
                    go.Violin(
                        x=[depth] * len(vals),
                        y=vals,
                        name=name,
                        marker_color=color,
                        legendgroup=group,
                        showlegend=False,
                        side=side,
                        width=0.4,
                    ),
                    row=row,
                    col=1,
                )


def create_route_comparison_figure(
    n1_stats: list[RouteStats],
    n5_stats: list[RouteStats],
) -> go.Figure:
    """
    Create a plotly figure comparing n1 and n5 route statistics.

    The figure has 3 rows:
    - Row 1: Bar chart of route counts by depth
    - Row 2: Box plots of target HAC by depth
    - Row 3: Box plots of target MW by depth

    Args:
        n1_stats: Statistics for n1 routes.
        n5_stats: Statistics for n5 routes.

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.12)
    N1_COLOR = "#5e548e"
    N5_COLOR = "#a53860"

    # Collect all depths for consistent x-axis
    all_depths = sorted(set(s.depth for s in n1_stats) | set(s.depth for s in n5_stats))

    # Row 1: Bar chart of counts by depth
    n1_counts = defaultdict(int)
    n5_counts = defaultdict(int)
    for s in n1_stats:
        n1_counts[s.depth] += 1
    for s in n5_stats:
        n5_counts[s.depth] += 1

    for counts, name, color in [
        (n1_counts, "n1 evaluation set", N1_COLOR),
        (n5_counts, "n5 evaluation set", N5_COLOR),
    ]:
        y_vals = [counts[d] for d in all_depths]
        fig.add_trace(
            go.Bar(
                x=all_depths,
                y=y_vals,
                name=name,
                marker_color=color,
                text=y_vals,
                textposition="outside",
            ),
            row=1,
            col=1,
        )

    # Row 2: Violin plots of HAC by depth
    _add_violin_traces(fig, n1_stats, n5_stats, all_depths, "target_hac", 2, N1_COLOR, N5_COLOR)

    # Row 3: Violin plots of MW by depth
    _add_violin_traces(fig, n1_stats, n5_stats, all_depths, "target_mw", 3, N1_COLOR, N5_COLOR)

    # Update layout
    fig.update_layout(
        height=900,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40),
    )

    fig.update_xaxes(title_text="Route Length", row=3, col=1)
    fig.update_yaxes(title_text="Count", range=[0, 5000], row=1, col=1)
    fig.update_yaxes(title_text="Heavy Atom Count", dtick=10, row=2, col=1)
    fig.update_yaxes(title_text="Molecular Weight (Da)", dtick=100, row=3, col=1)
    Styler().apply_style(fig)
    return fig
