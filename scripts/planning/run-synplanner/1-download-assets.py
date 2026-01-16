"""
Usage:
    uv run --directory scripts/planning/run-synplanner 1-download-assets.py
"""

from pathlib import Path

from synplan.utils.loading import download_selected_files

# download SynPlanner data
base_dir = Path(__file__).resolve().parents[3]
data_folder = base_dir / "data" / "0-assets" / "model-configs" / "synplanner"

assets = [
    ("uspto", "uspto_reaction_rules.pickle"),
    ("uspto/weights", "filtering_policy_network.ckpt"),
    ("uspto/weights", "ranking_policy_network.ckpt"),
    ("uspto/weights", "value_network.ckpt"),
]

download_selected_files(files_to_get=assets, save_to=data_folder, extract_zips=True)
