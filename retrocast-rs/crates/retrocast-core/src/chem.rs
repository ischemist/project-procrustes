use crate::{
    error::{EngineError, Result},
    schema::{CanonicalSmiles, InchiKey},
};

#[cxx::bridge(namespace = "retrocast")]
mod ffi {
    struct ChemResult {
        ok: bool,
        invalid_smiles: bool,
        smiles: String,
        inchikey: String,
        error: String,
    }

    struct DescriptorResult {
        ok: bool,
        invalid_smiles: bool,
        heavy_atom_count: u32,
        molecular_weight: f64,
        chiral_center_count: u32,
        error: String,
    }

    unsafe extern "C++" {
        include!("chem.hpp");
        fn canonicalize_smiles(
            smiles: &str,
            remove_mapping: bool,
            ignore_stereo: bool,
        ) -> ChemResult;
        fn inchi_key(smiles: &str, ignore_stereo: bool) -> ChemResult;
        fn molecular_descriptors(smiles: &str) -> DescriptorResult;
        fn rdkit_version() -> String;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InchiKeyLevel {
    Full,
    NoStereo,
    Connectivity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MolecularDescriptors {
    pub heavy_atom_count: u32,
    pub molecular_weight: f64,
    pub chiral_center_count: u32,
}

pub fn version() -> String {
    ffi::rdkit_version()
}

pub fn normalize(smiles: &str) -> Result<(CanonicalSmiles, InchiKey)> {
    normalize_with_mapping(smiles, false)
}

pub fn normalize_unmapped(smiles: &str) -> Result<(CanonicalSmiles, InchiKey)> {
    normalize_with_mapping(smiles, true)
}

pub fn canonicalize(
    smiles: &str,
    remove_mapping: bool,
    ignore_stereo: bool,
) -> Result<CanonicalSmiles> {
    let result = ffi::canonicalize_smiles(smiles, remove_mapping, ignore_stereo);
    if result.ok {
        Ok(result.smiles.try_into()?)
    } else {
        Err(chemistry_error(smiles, result.invalid_smiles, result.error))
    }
}

pub fn inchi_key(smiles: &str, level: InchiKeyLevel) -> Result<String> {
    let result = ffi::inchi_key(smiles, level == InchiKeyLevel::NoStereo);
    if !result.ok {
        return Err(chemistry_error(smiles, result.invalid_smiles, result.error));
    }
    if level == InchiKeyLevel::Connectivity {
        Ok(result.inchikey[..14].to_owned())
    } else {
        Ok(result.inchikey)
    }
}

pub fn descriptors(smiles: &str) -> Result<MolecularDescriptors> {
    let result = ffi::molecular_descriptors(smiles);
    if result.ok {
        Ok(MolecularDescriptors {
            heavy_atom_count: result.heavy_atom_count,
            molecular_weight: result.molecular_weight,
            chiral_center_count: result.chiral_center_count,
        })
    } else {
        Err(chemistry_error(smiles, result.invalid_smiles, result.error))
    }
}

pub fn reduce_inchi_key(inchikey: &str, level: InchiKeyLevel) -> Result<String> {
    let parts: Vec<&str> = inchikey.split('-').collect();
    if !matches!(parts.len(), 1 | 3) || parts[0].len() != 14 {
        return Err(EngineError::InvalidInchiKey {
            inchikey: inchikey.to_owned(),
            message: "expected a 14-character connectivity key or a three-block InChIKey"
                .to_owned(),
        });
    }

    if parts.len() == 1 && level != InchiKeyLevel::Connectivity {
        return Err(EngineError::InchiKeyUpscale {
            inchikey: inchikey.to_owned(),
            target_level: level.as_str(),
        });
    }

    match level {
        InchiKeyLevel::Full => Ok(inchikey.to_owned()),
        InchiKeyLevel::NoStereo => Ok(format!("{}-UHFFFAOYSA-{}", parts[0], parts[2])),
        InchiKeyLevel::Connectivity => Ok(parts[0].to_owned()),
    }
}

impl InchiKeyLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::NoStereo => "no_stereo",
            Self::Connectivity => "connectivity",
        }
    }
}

fn normalize_with_mapping(
    smiles: &str,
    remove_mapping: bool,
) -> Result<(CanonicalSmiles, InchiKey)> {
    let result = ffi::canonicalize_smiles(smiles, remove_mapping, false);
    if result.ok {
        Ok((result.smiles.try_into()?, result.inchikey.try_into()?))
    } else {
        Err(chemistry_error(smiles, result.invalid_smiles, result.error))
    }
}

fn chemistry_error(smiles: &str, invalid_smiles: bool, message: String) -> EngineError {
    if invalid_smiles {
        EngineError::InvalidSmiles {
            smiles: smiles.to_owned(),
            message,
        }
    } else {
        EngineError::Chemistry {
            smiles: smiles.to_owned(),
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InchiKeyLevel, canonicalize, descriptors, inchi_key, normalize, reduce_inchi_key};

    #[test]
    fn canonicalizes_smiles_and_calculates_inchikey_with_rdkit() {
        let identity = normalize("OCC").expect("ethanol is valid");
        assert_eq!(identity.0, "CCO");
        assert_eq!(identity.1, "LFQSCWFLJHTTHZ-UHFFFAOYSA-N");
    }

    #[test]
    fn rejects_invalid_smiles() {
        assert!(normalize("not-smiles").is_err());
    }

    #[test]
    fn supports_public_identifier_levels_and_descriptors() {
        let lactic_acid = "C[C@H](O)C(=O)O";
        assert_eq!(
            canonicalize(lactic_acid, false, true).unwrap(),
            "CC(O)C(=O)O"
        );
        assert_eq!(
            inchi_key(lactic_acid, InchiKeyLevel::NoStereo).unwrap(),
            "JVTAAEKCZFNVCJ-UHFFFAOYSA-N"
        );
        assert_eq!(
            inchi_key(lactic_acid, InchiKeyLevel::Connectivity).unwrap(),
            "JVTAAEKCZFNVCJ"
        );
        let values = descriptors(lactic_acid).unwrap();
        assert_eq!(values.heavy_atom_count, 6);
        assert!((values.molecular_weight - 90.031_694).abs() < 1e-6);
        assert_eq!(values.chiral_center_count, 1);
        assert_eq!(
            reduce_inchi_key("JVTAAEKCZFNVCJ-REOHCLBHSA-N", InchiKeyLevel::NoStereo).unwrap(),
            "JVTAAEKCZFNVCJ-UHFFFAOYSA-N"
        );
    }
}
