from typing import NewType

SmilesStr = NewType("SmilesStr", str)
InchiKeyStr = NewType("InchiKeyStr", str)
"""Represents a canonical SMILES string."""

ReactionSmilesStr = NewType("ReactionSmilesStr", str)
"""Represents a reaction SMILES string, e.g., 'reactant1.reactant2>>product'."""
