"""
Usage:
    uv run --directory scripts/planning/run-synplanner 1-download-assets.py
"""

from synplan.utils.loading import download_selected_files
from utils import get_synplanner_paths

paths = get_synplanner_paths()

assets = [
    ("uspto", "uspto_reaction_rules.pickle"),
    ("uspto/weights", "filtering_policy_network.ckpt"),
    ("uspto/weights", "ranking_policy_network.ckpt"),
    ("uspto/weights", "value_network.ckpt"),
]

download_selected_files(files_to_get=assets, save_to=paths.synplanner_dir, extract_zips=True)
