"""
visualization script for synthesis results.

loads benchmark data and generates performance plots based on a yaml config.
"""

import argparse
import logging
from pathlib import Path

from retrocast.analysis import performance
from retrocast.analysis.performance import VisualizationConfig

# --- constants ---
DEFAULT_VIZ_CONFIG_PATH = Path("data/analysis/performance-plots.yaml")


def main(config_path: Path):
    """main execution function."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not config_path.exists():
        logging.error(f"error: visualization config not found at {config_path}")
        return

    logging.info(f"loading visualization configuration from {config_path}...")
    viz_config: VisualizationConfig = performance.load_visualization_config(config_path)
    ps = viz_config.plot_settings  # convenience alias

    input_path = Path(ps.input_data_path)
    processed_path = Path(ps.processed_data_path)
    output_dir = Path(ps.output_dir)

    logging.info("discovering model names from manifests...")
    discovered_names = performance.discover_model_names(processed_path)
    logging.info(f"-> discovered {len(discovered_names)} models.")

    logging.info("building model display map...")
    model_display_map = performance.build_model_display_map(discovered_names, viz_config.models)
    logging.info(f"-> mapped {len(model_display_map)} models with display settings.")

    logging.info(f"\nloading data from {input_path}...")
    records = performance.load_benchmark_data(input_path)
    num_datasets = len(set(r.dataset for r in records))
    logging.info(f"found {len(records)} records across {num_datasets} datasets.")

    logging.info("\ngenerating plots...")

    # 1. generate the combined plot (one per dataset, stacked vertically)
    performance.plot_performance_summary(
        records=records,
        model_display_map=model_display_map,
        plot_settings=ps,
        output_dir=output_dir,
    )

    # 2. generate separate plots (one figure per dataset)
    performance.plot_performance_by_dataset(
        records=records,
        model_display_map=model_display_map,
        plot_settings=ps,
        output_dir=output_dir,
    )

    logging.info("\n...done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate performance plots.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_VIZ_CONFIG_PATH,
        help=f"path to the visualization config yaml file (default: {DEFAULT_VIZ_CONFIG_PATH})",
    )
    args = parser.parse_args()
    main(args.config)
