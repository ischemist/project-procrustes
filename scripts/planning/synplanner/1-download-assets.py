"""
Usage:
    uv run --extra synplanner scripts/synplanner/1-download-assets.py
"""

from pathlib import Path

from synplan.utils.loading import download_selected_files

# download SynPlanner data
data_folder = Path("data/0-assets/model-configs/synplanner").resolve()

assets = [
    ("uspto", "uspto_reaction_rules.pickle"),
    ("uspto/weights", "filtering_policy_network.ckpt"),
    ("uspto/weights", "ranking_policy_network.ckpt"),
    ("uspto/weights", "value_network.ckpt"),
]

download_selected_files(files_to_get=assets, save_to=data_folder, extract_zips=True)
