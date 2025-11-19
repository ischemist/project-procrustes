"""
Usage:
    uv run scripts/paroutes/3-export-reaction-smiles.py
"""

from pathlib import Path

from retrocast.export import extract_reactions_from_routes

from retrocast.curation import get_reaction_signatures
from retrocast.io import load_routes, save_reaction_smiles

# Load data
all_routes = load_routes(Path("data/paroutes/processed/all-routes.json.gz"))
n1_routes = load_routes(Path("data/paroutes/processed/n1-routes.json.gz"))
n5_routes = load_routes(Path("data/paroutes/processed/n5-routes.json.gz"))
# Get test set reaction signatures
test_reactions = get_reaction_signatures(n1_routes) | get_reaction_signatures(n5_routes)
# Extract reactions excluding test set
train_reactions = extract_reactions_from_routes(all_routes, exclude_reactions=test_reactions)
# Save
save_reaction_smiles(train_reactions, Path("output/train-reactions.txt.gz"))
