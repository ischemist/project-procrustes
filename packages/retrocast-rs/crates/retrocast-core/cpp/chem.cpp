#include "chem.hpp"

#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/inchi.h>
#include <RDGeneral/versions.h>

#include <exception>
#include <memory>
#include <string>

namespace retrocast {

namespace {

std::unique_ptr<RDKit::ROMol> parse_smiles(rust::Str input) {
  const std::string smiles(input);
  return smiles.empty()
             ? nullptr
             : std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

void set_parse_error(ChemResult& result, rust::Str input) {
  result.invalid_smiles = true;
  result.error = input.empty() ? "SMILES input must be non-empty"
                               : "invalid SMILES";
}

void set_parse_error(DescriptorResult& result, rust::Str input) {
  result.invalid_smiles = true;
  result.error = input.empty() ? "SMILES input must be non-empty"
                               : "invalid SMILES";
}

}  // namespace

ChemResult canonicalize_smiles(rust::Str input, bool remove_mapping,
                               bool ignore_stereo) {
  ChemResult result;
  try {
    std::unique_ptr<RDKit::ROMol> molecule = parse_smiles(input);
    if (!molecule) {
      set_parse_error(result, input);
      return result;
    }
    if (remove_mapping) {
      for (auto atom : molecule->atoms()) {
        atom->setAtomMapNum(0);
      }
    }
    result.smiles = RDKit::MolToSmiles(*molecule, !ignore_stereo, false);
    result.inchikey = RDKit::MolToInchiKey(*molecule);
    if (result.inchikey.empty()) {
      result.error = "RDKit returned an empty InChIKey";
      return result;
    }
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  } catch (...) {
    result.error = "unknown RDKit failure";
  }
  return result;
}

ChemResult inchi_key(rust::Str input, bool ignore_stereo) {
  ChemResult result;
  try {
    std::unique_ptr<RDKit::ROMol> molecule = parse_smiles(input);
    if (!molecule) {
      set_parse_error(result, input);
      return result;
    }

    RDKit::ExtraInchiReturnValues details;
    const char* options = ignore_stereo ? "-SNon" : nullptr;
    const std::string inchi = RDKit::MolToInchi(*molecule, details, options);
    // InChI uses return code 1 for a usable result with warnings, such as an
    // unspecified stereocenter. Python RDKit's MolToInchiKey accepts that
    // result too; only codes 2 and above are failures.
    if (details.returnCode >= 2) {
      result.error = "InChI generation failed with code " +
                     std::to_string(details.returnCode) + ": " +
                     details.messagePtr;
      return result;
    }
    result.inchikey = RDKit::InchiToInchiKey(inchi);
    if (result.inchikey.empty()) {
      result.error = "RDKit returned an empty InChIKey";
      return result;
    }
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  } catch (...) {
    result.error = "unknown RDKit failure";
  }
  return result;
}

DescriptorResult molecular_descriptors(rust::Str input) {
  DescriptorResult result;
  try {
    std::unique_ptr<RDKit::ROMol> molecule = parse_smiles(input);
    if (!molecule) {
      set_parse_error(result, input);
      return result;
    }

    result.heavy_atom_count = molecule->getNumAtoms();
    result.molecular_weight = RDKit::Descriptors::calcExactMW(*molecule);
    for (const auto atom : molecule->atoms()) {
      if (atom->hasProp("_CIPCode")) {
        ++result.chiral_center_count;
      }
    }
    result.ok = true;
  } catch (const std::exception& error) {
    result.error = error.what();
  } catch (...) {
    result.error = "unknown RDKit failure";
  }
  return result;
}

rust::String rdkit_version() { return RDKit::rdkitVersion; }

}  // namespace retrocast
