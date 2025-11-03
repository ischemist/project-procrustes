"""
visualization script for synthesis results.

this script loads benchmark data and generates performance plots.
configuration is handled directly in this file.
"""

from pathlib import Path

from ursa.analysis import performance

# --- configuration ---
# all your settings live here. no magic numbers scattered in the code.

DATA_PATH = Path("data/bench-results-2025-10-30.csv")
OUTPUT_DIR = Path("data/analysis")
PROCESSED_DATA_PATH = Path("data/processed")

# fill this with whatever abbreviations you want.
# the key should be the 'model id' from the csv.
MODEL_ABBREVIATIONS = {
    "22f978fe": "dms-wide",
    "abcdef12": "baseline-sm",
    "12345678": "expert-xl",
    # ... add all your model ids and desired abbreviations here
}


def main():
    """main execution function."""
    if not DATA_PATH.exists():
        print(f"error: data file not found at {DATA_PATH}")
        return
    print("discovering model names from manifests...")
    discovered_names = performance.discover_model_names(PROCESSED_DATA_PATH)
    print(f"-> discovered {len(discovered_names)} models.")

    # combine discovered names with manual abbreviations. manual ones win.
    final_model_map = discovered_names.copy()
    final_model_map.update(MODEL_ABBREVIATIONS)

    print(f"loading data from {DATA_PATH}...")
    df = performance.load_benchmark_data(DATA_PATH)
    print("data loaded successfully.")
    print(f"found {len(df)} records across {df['dataset'].nunique()} datasets.")

    print("\ngenerating plots...")

    # 1. generate the combined plot (3 rows, 1 figure)
    performance.plot_performance_summary(
        df=df,
        model_abbreviations=final_model_map,
        output_dir=OUTPUT_DIR,
    )

    # 2. generate separate plots (one figure per dataset)
    performance.plot_performance_by_dataset(
        df=df,
        model_abbreviations=final_model_map,
        output_dir=OUTPUT_DIR,
    )

    print("\n...done.")


if __name__ == "__main__":
    main()
