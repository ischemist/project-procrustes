import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from retrocast.exceptions import SyntheseusSerializationError, TtlRetroSerializationError


def _build_tree_recursive(
    smiles: str,
    smiles_to_ornode: dict[str, Any],
    product_smiles_to_andnode: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively builds a JSON-serializable dictionary from the pre-processed graph nodes.

    This function traverses the route from a given molecule SMILES, reconstructing
    the tree structure required for the benchmark format.
    """
    if smiles not in smiles_to_ornode:
        raise SyntheseusSerializationError(f"Incomplete route graph: OrNode for SMILES '{smiles}' not found.")

    or_node = smiles_to_ornode[smiles]
    is_purchasable = or_node.mol.metadata.get("is_purchasable", False)

    # Base case: molecule is purchasable or has no reaction leading to it
    is_leaf = is_purchasable or smiles not in product_smiles_to_andnode

    mol_dict: dict[str, Any] = {
        "smiles": smiles,
        "type": "mol",
        "in_stock": is_purchasable,
        "children": [],
    }

    if not is_leaf:
        and_node = product_smiles_to_andnode[smiles]

        # Build the reaction dict
        reactant_smiles = sorted([r.smiles for r in and_node.reaction.reactants])
        product_smiles = next(iter(and_node.reaction.products)).smiles
        rxn_smiles_str = f"{'.'.join(reactant_smiles)}>>{product_smiles}"

        reaction_dict: dict[str, Any] = {"smiles": rxn_smiles_str, "type": "reaction", "children": []}

        # Recurse on reactants
        for reactant_mol in and_node.reaction.reactants:
            reactant_tree = _build_tree_recursive(reactant_mol.smiles, smiles_to_ornode, product_smiles_to_andnode)
            reaction_dict["children"].append(reactant_tree)

        mol_dict["children"].append(reaction_dict)

    return mol_dict


def serialize_route(route_nodes: Iterable[Any], target_smiles: str) -> dict[str, Any]:
    """
    Serializes a single syntheseus route (a collection of nodes) into a JSON-compatible dict.

    Args:
        route_nodes: A collection containing the AndNode and OrNode objects of a single route.
        target_smiles: The SMILES string of the root target molecule to start traversal.

    Returns:
        A nested dictionary representing the retrosynthetic tree.
    """
    smiles_to_ornode = {node.mol.smiles: node for node in route_nodes if hasattr(node, "mol")}
    product_smiles_to_andnode = {
        next(iter(node.reaction.products)).smiles: node for node in route_nodes if hasattr(node, "reaction")
    }

    if target_smiles not in smiles_to_ornode:
        raise SyntheseusSerializationError(f"Target SMILES '{target_smiles}' not found in the provided route nodes.")

    return _build_tree_recursive(target_smiles, smiles_to_ornode, product_smiles_to_andnode)


def serialize_and_save(
    routes_by_target: dict[str, list[Iterable[Any]]],
    output_path: Path,
) -> None:
    """
    Serializes a dictionary of syntheseus routes to a gzipped JSON file.

    This is the main entry point function.

    Args:
        routes_by_target: A dictionary mapping target SMILES to a list of routes,
                          where each route is a collection of nodes.
        output_path: The path to the output .json.gz file.
    """
    serialized_data = {}
    for target_id, routes in routes_by_target.items():
        serialized_routes = []
        # We need the actual target SMILES to start the traversal, assuming the key is not the smiles.
        # This part might need adjustment depending on what `routes_by_target` keys are.
        # For now, let's assume the first OrNode at depth 0 is the target.
        # A more robust way is to pass in a target_id -> target_smiles map.
        # Let's assume for now the key IS the target smiles.
        target_smiles = target_id
        for i, route_nodes in enumerate(routes):
            try:
                tree_dict = serialize_route(route_nodes, target_smiles)
                serialized_routes.append(tree_dict)
            except SyntheseusSerializationError as e:
                print(f"Warning: Could not serialize route {i} for target {target_smiles}. Reason: {e}")

        # We use the raw target smiles/id as the key, retrocast will map it to an ID later.
        serialized_data[target_id] = serialized_routes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(serialized_data, indent=2)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        f.write(json_str)


def _get_multistepttl_advanced_scores(tree: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """calculates forward confidence score for each route."""
    scores = []
    for route in tree["Route"]:
        prod = 1.0
        try:
            for rxn_id in route:
                prod *= predictions.loc[rxn_id, "Prob_Forward_Prediction_1"]
            scores.append(prod)
        except KeyError as e:
            # this can happen if a route in tree.pkl references a reaction not in predictions.pkl
            raise TtlRetroSerializationError(f"reaction id {e} from route not found in predictions.") from e
    tree["fwd_conf_score"] = scores
    return tree


def serialize_multistepttl_target(tree_df: pd.DataFrame, predictions_df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    serializes multistepttl dataframes for a single target into a json-compatible list of routes.

    args:
        tree_df: the dataframe from `*__tree.pkl`.
        predictions_df: the dataframe from `*__prediction.pkl`.

    returns:
        a list of dictionaries, where each dictionary represents a solved route.

    raises:
        TtlRetroSerializationError: if data is inconsistent.
    """
    if "index" in predictions_df.columns and predictions_df.index.name != "index":
        predictions_df = predictions_df.set_index("index")

    solved_routes_df = tree_df[tree_df["Solved"] == "Yes"].copy()
    if solved_routes_df.empty:
        return []

    solved_routes_df = _get_multistepttl_advanced_scores(solved_routes_df, predictions_df)

    output_routes = []
    for _, route_row in solved_routes_df.iterrows():
        reactions = []
        route_rxn_ids = route_row["Route"]
        for rxn_id in route_rxn_ids:
            try:
                pred_row = predictions_df.loc[rxn_id]
                reactants = pred_row["Retro"].split(".")
                product = pred_row["Target"]
                reactions.append({"product": product, "reactants": reactants})
            except KeyError as e:
                # this should be caught by the score calculation, but as a safeguard:
                raise TtlRetroSerializationError(f"reaction id {rxn_id} from route not found in predictions.") from e

        output_routes.append(
            {
                "reactions": reactions,
                "metadata": {
                    "fwd_conf_score": route_row.get("fwd_conf_score"),
                    "score": route_row.get("Score"),
                    "steps": len(route_rxn_ids),
                },
            }
        )

    return output_routes


def serialize_multistepttl_directory(target_dir: Path) -> list[dict[str, Any]] | None:
    """
    finds pickles in a directory, loads them, and serializes the routes.

    args:
        target_dir: the directory containing the `*__tree.pkl` and `*__prediction.pkl` files.

    returns:
        the serialized route data as a list of dicts, or none if files are not found.
    """

    try:
        tree_pkl = next(target_dir.glob("*__tree.pkl"))
        predictions_pkl = next(target_dir.glob("*__prediction.pkl"))

        tree_df = pd.read_pickle(tree_pkl)
        predictions_df = pd.read_pickle(predictions_pkl)

        return serialize_multistepttl_target(tree_df, predictions_df)

    except (StopIteration, FileNotFoundError):
        # logger should handle this in the calling script
        return None
    except Exception as e:
        raise TtlRetroSerializationError(f"failed to process pickles in {target_dir}: {e}") from e
