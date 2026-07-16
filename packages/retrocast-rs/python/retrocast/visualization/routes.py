from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from retrocast.chem import get_chiral_center_count, get_heavy_atom_count, get_molecular_weight
from retrocast.models.route import MoleculeView, Route
from retrocast.visualization.theme import apply_layout


@dataclass(frozen=True, slots=True)
class RouteStats:
    depth: int
    target_hac: int
    target_mw: float
    target_chiral: int
    is_convergent: bool


def extract_route_stats(routes: dict[str, Route]) -> list[RouteStats]:
    stats = []
    for route in routes.values():
        smiles = route.target.smiles
        stats.append(
            RouteStats(
                depth=route.depth(),
                target_hac=get_heavy_atom_count(smiles),
                target_mw=get_molecular_weight(smiles),
                target_chiral=get_chiral_center_count(smiles),
                is_convergent=is_convergent_route(route),
            )
        )
    return stats


def is_convergent_route(route: Route) -> bool:
    return any(len(branching_children(molecule)) > 1 for molecule in iter_internal_molecules(route))


def iter_internal_molecules(route: Route) -> list[MoleculeView]:
    stack = [route.molecule_at("rc:m:/")]
    internal = []
    while stack:
        molecule = stack.pop()
        reaction = molecule.produced_by()
        if reaction is None:
            continue
        internal.append(molecule)
        stack.extend(reaction.reactants())
    return internal


def branching_children(molecule: MoleculeView) -> list[MoleculeView]:
    reaction = molecule.produced_by()
    if reaction is None:
        return []
    return [reactant for reactant in reaction.reactants() if reactant.produced_by() is not None]


def create_route_comparison_figure(
    n1_stats: list[RouteStats],
    n5_stats: list[RouteStats],
) -> go.Figure:
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.03)
    n1_color = "#5e548e"
    n5_color = "#a53860"
    depths = sorted({item.depth for item in n1_stats} | {item.depth for item in n5_stats})

    add_count_bars(fig, n1_stats, n5_stats, depths, n1_color, n5_color, row=1, convergent_only=False)
    add_count_bars(fig, n1_stats, n5_stats, depths, n1_color, n5_color, row=2, convergent_only=True)
    add_violin_traces(fig, n1_stats, n5_stats, depths, "target_hac", n1_color, n5_color, row=3)
    add_violin_traces(fig, n1_stats, n5_stats, depths, "target_mw", n1_color, n5_color, row=4)
    add_violin_traces(
        fig,
        [item for item in n1_stats if item.target_chiral > 0],
        [item for item in n5_stats if item.target_chiral > 0],
        depths,
        "target_chiral",
        n1_color,
        n5_color,
        row=5,
    )

    apply_layout(fig, height=1400)
    fig.update_layout(barmode="group", margin={"t": 40})
    for row in range(1, 6):
        fig.update_xaxes(title_text="Route Length" if row == 5 else None, range=[1.5, 10.5], row=row, col=1)

    title_standoff = 10
    n1_total = count_by_depth(n1_stats, convergent_only=False)
    n5_total = count_by_depth(n5_stats, convergent_only=False)
    n1_convergent = count_by_depth(n1_stats, convergent_only=True)
    n5_convergent = count_by_depth(n5_stats, convergent_only=True)
    max_total = max([n1_total[depth] for depth in depths] + [n5_total[depth] for depth in depths], default=0)
    max_convergent = max(
        [n1_convergent[depth] for depth in depths] + [n5_convergent[depth] for depth in depths],
        default=0,
    )
    fig.update_yaxes(title_text="Total Count", title_standoff=title_standoff, range=[0, max_total * 1.15], row=1, col=1)
    fig.update_yaxes(
        title_text="Convergent Count",
        title_standoff=title_standoff,
        range=[0, max_convergent * 1.15],
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Heavy Atom Count", title_standoff=title_standoff, dtick=10, row=3, col=1)
    fig.update_yaxes(title_text="Molecular Weight (Da)", title_standoff=title_standoff, dtick=100, row=4, col=1)
    fig.update_yaxes(title_text="Chiral Centers", title_standoff=title_standoff, dtick=2, row=5, col=1)
    return fig


def add_count_bars(
    fig: go.Figure,
    n1_stats: list[RouteStats],
    n5_stats: list[RouteStats],
    depths: list[int],
    n1_color: str,
    n5_color: str,
    *,
    row: int,
    convergent_only: bool,
) -> None:
    n1_counts = count_by_depth(n1_stats, convergent_only=convergent_only)
    n5_counts = count_by_depth(n5_stats, convergent_only=convergent_only)
    showlegend = not convergent_only
    fig.add_trace(
        go.Bar(
            x=depths,
            y=[n1_counts[depth] for depth in depths],
            name="n1 evaluation set" if showlegend else "n1 convergent",
            marker_color=n1_color,
            legendgroup="n1",
            showlegend=showlegend,
            text=[n1_counts[depth] for depth in depths],
            textposition="outside",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=depths,
            y=[n5_counts[depth] for depth in depths],
            name="n5 evaluation set" if showlegend else "n5 convergent",
            marker_color=n5_color,
            legendgroup="n5",
            showlegend=showlegend,
            text=[n5_counts[depth] for depth in depths],
            textposition="outside",
        ),
        row=row,
        col=1,
    )


def add_violin_traces(
    fig: go.Figure,
    n1_stats: list[RouteStats],
    n5_stats: list[RouteStats],
    depths: list[int],
    field: str,
    n1_color: str,
    n5_color: str,
    *,
    row: int,
) -> None:
    for stats, name, color, side in ((n1_stats, "n1", n1_color, "negative"), (n5_stats, "n5", n5_color, "positive")):
        values_by_depth = values_for_depths(stats, depths, field)
        x_values = [depth for depth, values in values_by_depth.items() for _ in values]
        y_values = [value for values in values_by_depth.values() for value in values]
        if y_values:
            fig.add_trace(
                go.Violin(
                    x=x_values,
                    y=y_values,
                    name=f"{name} set",
                    marker_color=color,
                    legendgroup=name,
                    side=side,
                    width=0.4,
                    showlegend=False,
                ),
                row=row,
                col=1,
            )


def count_by_depth(stats: list[RouteStats], *, convergent_only: bool) -> Counter[int]:
    return Counter(item.depth for item in stats if item.is_convergent or not convergent_only)


def values_for_depths(stats: list[RouteStats], depths: list[int], field: str) -> dict[int, list[int | float]]:
    return {depth: [getattr(item, field) for item in stats if item.depth == depth] for depth in depths}
