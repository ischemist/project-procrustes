"""
Usage:
    uv run --extra synplanner --extra torch-cpu scripts/synplanner/1-download-assets.py
"""

from pathlib import Path

from synplan.utils.loading import download_all_data

# download SynPlanner data
data_folder = Path("data/models/synplanner").resolve()
download_all_data(save_to=data_folder)
