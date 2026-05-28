from typing import NewType

# Type definitions for chemical identifiers and structures
SmilesStr = NewType("SmilesStr", str)
"""Represents a canonical SMILES string for a molecule."""

InChIKeyStr = NewType("InChIKeyStr", str)
"""Represents an InChIKey string, the primary canonical identifier for molecules."""

InchiKeyStr = InChIKeyStr
"""Deprecated spelling. Use InChIKeyStr."""

ReactionSmilesStr = NewType("ReactionSmilesStr", str)
"""Represents a reaction SMILES string, e.g., 'reactant1.reactant2>>product'."""

ErrorCode = NewType("ErrorCode", str)
"""Stable machine-readable error code. Messages remain human-readable prose."""
