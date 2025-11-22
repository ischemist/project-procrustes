"""Route analysis and visualization functions."""

from collections import defaultdict
from dataclasses import dataclass

import plotly.graph_objects as go
from ischemist.plotly import Styler
from plotly.subplots import make_subplots

from retrocast.chem import get_chiral_center_count, get_heavy_atom_count, get_molecular_weight
from retrocast.models.chem import Route


@dataclass
class RouteStats:
    """Statistics for a single route."""

    depth: int
    target_hac: int
    target_mw: float
    target_chiral: int
    is_convergent: bool


def extract_route_stats(routes: dict[str, Route]) -> list[RouteStats]:
    """
    Extract statistics from routes.

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.

    Returns:
        List of RouteStats, one per route.
    """
    stats = []
    for route in routes.values():
        smiles = route.target.smiles
        stats.append(
            RouteStats(
                depth=route.length,
                target_hac=get_heavy_atom_count(smiles),
                target_mw=get_molecular_weight(smiles),
                target_chiral=get_chiral_center_count(smiles),
                is_convergent=route.has_convergent_reaction,
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

    The figure has 5 rows:
    - Row 1: Bar chart of total route counts by depth
    - Row 2: Bar chart of convergent route counts by depth
    - Row 3: Violin plots of target HAC by depth
    - Row 4: Violin plots of target MW by depth
    - Row 5: Violin plots of target chiral center count by depth

    Args:
        n1_stats: Statistics for n1 routes.
        n5_stats: Statistics for n5 routes.

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.03)
    N1_COLOR = "#5e548e"
    N5_COLOR = "#a53860"

    # Collect all depths for consistent x-axis
    all_depths = sorted(set(s.depth for s in n1_stats) | set(s.depth for s in n5_stats))

    # Count total and convergent routes by depth
    n1_total = defaultdict(int)
    n1_convergent = defaultdict(int)
    n5_total = defaultdict(int)
    n5_convergent = defaultdict(int)

    for s in n1_stats:
        n1_total[s.depth] += 1
        if s.is_convergent:
            n1_convergent[s.depth] += 1

    for s in n5_stats:
        n5_total[s.depth] += 1
        if s.is_convergent:
            n5_convergent[s.depth] += 1

    # Row 1: Bar chart of total route counts
    n1_total_vals = [n1_total[d] for d in all_depths]
    n5_total_vals = [n5_total[d] for d in all_depths]

    fig.add_trace(
        go.Bar(
            x=all_depths,
            y=n1_total_vals,
            name="n1 evaluation set",
            marker_color=N1_COLOR,
            legendgroup="n1",
            text=n1_total_vals,
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=all_depths,
            y=n5_total_vals,
            name="n5 evaluation set",
            marker_color=N5_COLOR,
            legendgroup="n5",
            text=n5_total_vals,
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # Row 2: Bar chart of convergent route counts
    n1_conv_vals = [n1_convergent[d] for d in all_depths]
    n5_conv_vals = [n5_convergent[d] for d in all_depths]

    fig.add_trace(
        go.Bar(
            x=all_depths,
            y=n1_conv_vals,
            name="n1 convergent",
            marker_color=N1_COLOR,
            legendgroup="n1",
            showlegend=False,
            text=n1_conv_vals,
            textposition="outside",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=all_depths,
            y=n5_conv_vals,
            name="n5 convergent",
            marker_color=N5_COLOR,
            legendgroup="n5",
            showlegend=False,
            text=n5_conv_vals,
            textposition="outside",
        ),
        row=2,
        col=1,
    )

    # Row 3: Violin plots of HAC by depth
    _add_violin_traces(fig, n1_stats, n5_stats, all_depths, "target_hac", 3, N1_COLOR, N5_COLOR)

    # Row 4: Violin plots of MW by depth
    _add_violin_traces(fig, n1_stats, n5_stats, all_depths, "target_mw", 4, N1_COLOR, N5_COLOR)

    # Row 5: Violin plots of chiral center count by depth (exclude zeros)
    n1_stats_chiral = [s for s in n1_stats if s.target_chiral > 0]
    n5_stats_chiral = [s for s in n5_stats if s.target_chiral > 0]
    _add_violin_traces(fig, n1_stats_chiral, n5_stats_chiral, all_depths, "target_chiral", 5, N1_COLOR, N5_COLOR)

    # Update layout
    fig.update_layout(
        height=1400,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40),
    )

    # Set x-axis range to always show route lengths 2-10 on all rows
    for row in range(1, 6):
        fig.update_xaxes(title_text="Route Length" if row == 5 else None, range=[1.5, 10.5], row=row, col=1)

    # Set consistent title standoff for y-axis alignment
    title_standoff = 10

    # Rows 1-2: Bar charts - adjust range to prevent text cutoff
    max_total = max(max(n1_total_vals, default=0), max(n5_total_vals, default=0))
    max_conv = max(max(n1_conv_vals, default=0), max(n5_conv_vals, default=0))

    fig.update_yaxes(
        title_text="Total Count",
        title_standoff=title_standoff,
        range=[0, max_total * 1.15],  # Add 15% padding for text labels
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Convergent Count",
        title_standoff=title_standoff,
        range=[0, max_conv * 1.15],  # Add 15% padding for text labels
        row=2,
        col=1,
    )

    # Rows 3-5: Violin plots - use consistent standoff
    fig.update_yaxes(title_text="Heavy Atom Count", title_standoff=title_standoff, dtick=10, row=3, col=1)
    fig.update_yaxes(title_text="Molecular Weight (Da)", title_standoff=title_standoff, dtick=100, row=4, col=1)
    fig.update_yaxes(title_text="Chiral Centers", title_standoff=title_standoff, dtick=2, row=5, col=1)
    Styler().apply_style(fig)
    return fig
