from typing import NewType

# Type definitions for chemical identifiers and structures
SmilesStr = NewType("SmilesStr", str)
"""Represents a canonical SMILES string for a molecule."""

InchiKeyStr = NewType("InchiKeyStr", str)
"""Represents an InChIKey string, the primary canonical identifier for molecules."""

ReactionSmilesStr = NewType("ReactionSmilesStr", str)
"""Represents a reaction SMILES string, e.g., 'reactant1.reactant2>>product'."""
