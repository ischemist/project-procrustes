#pragma once

#include "rust/cxx.h"
#include "retrocast-core/src/chem.rs.h"

namespace retrocast {

ChemResult canonicalize_smiles(rust::Str smiles, bool remove_mapping,
                               bool ignore_stereo);
ChemResult inchi_key(rust::Str smiles, bool ignore_stereo);
DescriptorResult molecular_descriptors(rust::Str smiles);
rust::String rdkit_version();

}  // namespace retrocast
