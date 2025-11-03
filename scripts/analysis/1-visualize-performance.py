"""
visualization script for synthesis results.

this script loads benchmark data and generates performance plots.
configuration is loaded from data/visualization-config.yaml
"""

from pathlib import Path

from ursa.analysis import performance

# --- configuration paths ---
VIZ_CONFIG_PATH = Path("data/analysis/performance-plots.yaml")
DATA_PATH = Path("data/bench-results-2025-10-30.csv")
PROCESSED_DATA_PATH = Path("data/processed")


def main():
    """main execution function."""
    if not DATA_PATH.exists():
        print(f"error: data file not found at {DATA_PATH}")
        return

    if not VIZ_CONFIG_PATH.exists():
        print(f"error: visualization config not found at {VIZ_CONFIG_PATH}")
        return

    print("loading visualization configuration...")
    viz_config = performance.load_visualization_config(VIZ_CONFIG_PATH)
    print(f"-> loaded config from {VIZ_CONFIG_PATH}")

    print("discovering model names from manifests...")
    discovered_names = performance.discover_model_names(PROCESSED_DATA_PATH)
    print(f"-> discovered {len(discovered_names)} models.")
    print("building model display map...")
    model_display_map = performance.build_model_display_map(discovered_names, viz_config)
    print(f"-> mapped {len(model_display_map)} models with display settings.")

    print(f"\nloading data from {DATA_PATH}...")
    df = performance.load_benchmark_data(DATA_PATH)
    print("data loaded successfully.")
    print(f"found {len(df)} records across {df['dataset'].nunique()} datasets.")

    # get plot settings from config
    plot_settings = viz_config.get("plot_settings", {})
    output_dir = Path(plot_settings.get("output_dir"))
    shared_xaxes = plot_settings.get("combined_figure", {}).get("shared_xaxes", True)
    shared_yaxes = plot_settings.get("combined_figure", {}).get("shared_yaxes", False)

    print("\ngenerating plots...")

    # 1. generate the combined plot (one per dataset, stacked vertically)
    performance.plot_performance_summary(
        df=df,
        model_display_map=model_display_map,
        output_dir=output_dir,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        viz_config=viz_config,
    )

    # 2. generate separate plots (one figure per dataset)
    performance.plot_performance_by_dataset(
        df=df,
        model_display_map=model_display_map,
        output_dir=output_dir,
        viz_config=viz_config,
    )

    print("\n...done.")


if __name__ == "__main__":
    main()
