"""Functions for exporting routes to various formats."""

from tqdm import tqdm

from retrocast.schemas import Molecule, ReactionSignature, Route


def collect_reaction_smiles_and_signatures(
    route: Route,
) -> list[tuple[str, ReactionSignature]]:
    reactions: list[tuple[str, ReactionSignature]] = []

    def _collect_reactions(node: Molecule) -> None:
        if node.is_leaf:
            return

        assert node.synthesis_step is not None

        step = node.synthesis_step

        # Build reaction SMILES: reactants>reagents>product
        reactant_smiles = sorted(r.smiles for r in step.reactants)
        reactant_part = ".".join(reactant_smiles)

        if step.reagents:
            reagent_part = ".".join(sorted(step.reagents))
        else:
            reagent_part = ""

        product_part = node.smiles
        rxn_smiles = f"{reactant_part}>{reagent_part}>{product_part}"

        # Build signature
        reactant_keys = frozenset(r.inchikey for r in step.reactants)
        sig: ReactionSignature = (reactant_keys, node.inchikey)

        reactions.append((rxn_smiles, sig))

        # Recursively collect from reactants
        for reactant in step.reactants:
            _collect_reactions(reactant)

    _collect_reactions(route.target)
    return reactions


def extract_reactions_from_routes(
    routes: dict[str, list[Route]],
    exclude_reactions: set[ReactionSignature] | None = None,
) -> set[str]:
    """
    Extract all unique reaction SMILES from a collection of routes.

    Args:
        routes: Dictionary mapping target IDs to lists of Route objects.
        exclude_reactions: Optional set of ReactionSignatures to exclude.

    Returns:
        Set of unique reaction SMILES strings.
    """
    if exclude_reactions is None:
        exclude_reactions = set()

    all_reactions: set[str] = set()

    for route_list in tqdm(routes.values(), desc="Extracting reactions", total=len(routes), dynamic_ncols=True):
        for route in route_list:
            for rxn_smiles, sig in collect_reaction_smiles_and_signatures(route):
                if sig not in exclude_reactions:
                    all_reactions.add(rxn_smiles)

    return all_reactions
