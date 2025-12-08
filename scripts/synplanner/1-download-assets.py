"""
Usage:
    uv run --extra synplanner scripts/synplanner/1-download-assets.py
"""

from pathlib import Path

from synplan.utils.loading import download_all_data

# download SynPlanner data
data_folder = Path("data/0-assets/model-configs/synplanner").resolve()
download_all_data(save_to=data_folder)
