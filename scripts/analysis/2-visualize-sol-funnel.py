import logging
from pathlib import Path

from retrocast.analysis.performance import (
    build_model_display_map,
    discover_model_names,
    load_benchmark_data,
    load_visualization_config,
    plot_prediction_funnel,
    plot_prediction_funnel_summary,
)

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- configuration ---
CONFIG_PATH = Path("data/analysis/performance-plots.yaml")


def main():
    """main orchestration function."""
    logging.info(f"loading configuration from {CONFIG_PATH}")
    config = load_visualization_config(CONFIG_PATH)
    settings = config.plot_settings
    bar_settings = config.bar_settings
    output_dir = Path(settings.output_dir)

    logging.info("loading and processing data...")
    model_id_to_name = discover_model_names(Path(settings.processed_data_path))
    records = load_benchmark_data(Path(settings.input_data_path))
    model_display_map = build_model_display_map(model_id_to_name, config.models)

    # generate a plot for each dataset individually
    logging.info("generating individual prediction funnel plots...")
    plot_prediction_funnel(
        records=records,
        model_display_map=model_display_map,
        bar_settings=bar_settings,
        dataset_totals=settings.dataset_totals,
        output_dir=output_dir,
    )

    # generate the combined summary plot
    logging.info("generating combined prediction funnel plot...")
    plot_prediction_funnel_summary(
        records=records,
        model_display_map=model_display_map,
        bar_settings=bar_settings,
        dataset_totals=settings.dataset_totals,
        output_dir=output_dir,
    )

    logging.info("...done.")


if __name__ == "__main__":
    main()
