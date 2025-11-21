"""
Runs a pairwise statistical tournament.
Generates a matrix showing the difference between every pair of models.

Usage:
    uv run scripts/05-tournament.py --benchmark stratified-linear-600-seed=42 --models dms-flash dms-wide dms-deep dms-flash-20M dms-explorer-xl dms-flex-duo
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from retrocast.io.data import BenchmarkResultsLoader
from retrocast.metrics.bootstrap import get_is_solvable, make_get_top_k
from retrocast.metrics.ranking import compute_pairwise_tournament
from retrocast.models.stats import ModelComparison
from retrocast.utils.logging import configure_script_logging, logger
from retrocast.visualization import plots

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

console = Console()


def display_tournament_matrix(comparisons: list[ModelComparison], model_names: list[str]) -> None:
    """
    Prints a text-based matrix of the tournament results.
    Rows = Model A, Columns = Model B. Cell = (A - B).
    """
    table = Table(title="Tournament Results (Row - Col)", box=None, show_lines=True, header_style="bold")

    # Create Header Row
    table.add_column("Model", style="bold cyan")
    for m in model_names:
        table.add_column(m, justify="center")

    # Create Matrix Data
    # Map (model_a, model_b) -> Comparison object
    comp_map = {}
    for c in comparisons:
        comp_map[(c.model_a, c.model_b)] = c

    for row_model in model_names:
        row_cells = [row_model]

        for col_model in model_names:
            if row_model == col_model:
                row_cells.append("[dim]-[/]")
                continue

            comp = comp_map.get((row_model, col_model))
            if not comp:
                row_cells.append("?")
                continue

            # Logic: Row - Col
            val = comp.diff_mean

            if not comp.is_significant:
                # Insignificant = dim text
                txt = f"[dim]{val:+.1%}[/]"
            else:
                # Significant
                if val > 0:
                    # Row is better (Green)
                    txt = f"[bold green]{val:+.1%}[/]"
                else:
                    # Row is worse (Red)
                    txt = f"[bold red]{val:+.1%}[/]"

            row_cells.append(txt)

        table.add_row(*row_cells)

    console.print(table)
    console.print("[dim]* Green = Row beats Col significantly. Red = Row loses to Col significantly.[/]\n")


def main():
    configure_script_logging(use_rich=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--stock", default="n5-stock")
    parser.add_argument("--metric", default="top-1", choices=["top-1", "solvability"])
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    loader = BenchmarkResultsLoader(DATA_DIR)

    # 1. Load Data
    loaded_models = {}

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Loading models...", total=len(args.models))

        for model in args.models:
            res = loader.load_evaluation(args.benchmark, model, args.stock)
            if res:
                loaded_models[model] = res
            progress.advance(task)

    if len(loaded_models) < 2:
        logger.error("[bold red]Need at least 2 valid models for a tournament.[/]")
        return

    # 2. Config Metric
    if args.metric == "solvability":
        extractor = get_is_solvable
        label = "Solvability"
    else:
        extractor = make_get_top_k(1)
        label = "Top-1 Accuracy"

    # 3. Run Tournament
    logger.info(f"Running tournament on [bold]{label}[/] (N={args.n_boot})...")

    # Computation can be slow, show a spinner
    with console.status("[bold green]Computing pairwise statistics...[/]"):
        results = compute_pairwise_tournament(loaded_models, extractor, label, n_boot=args.n_boot)

    # 4. Display Results
    # Extract sorted unique model names for the matrix headers
    unique_models = sorted(list(loaded_models.keys()))
    display_tournament_matrix(results, unique_models)

    # 5. Plot HTML
    output_dir = DATA_DIR / "6-comparisons" / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plots.plot_pairwise_matrix(results, label)
    out_file = output_dir / f"pairwise_matrix_{args.metric}.html"

    fig.write_html(out_file, include_plotlyjs="cdn", auto_open=True)

    logger.info(f"Interactive matrix saved to: [underline]{out_file}[/]")


if __name__ == "__main__":
    main()
