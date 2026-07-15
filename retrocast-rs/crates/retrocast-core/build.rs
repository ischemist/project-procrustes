use std::{env, path::PathBuf};

fn main() {
    let root = env::var_os("RDKIT_ROOT")
        .map(PathBuf::from)
        .or_else(default_rdkit_root)
        .expect("RDKit C++ installation not found; set RDKIT_ROOT");
    let include = env::var_os("RDKIT_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            first_existing(&[
                root.join("include/rdkit"),
                root.join("Library/include/rdkit"),
            ])
        });
    let lib = env::var_os("RDKIT_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| first_existing(&[root.join("lib"), root.join("Library/lib")]));
    let boost_root = env::var_os("BOOST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.clone());
    let boost_include = first_containing(
        &[
            boost_root.join("include"),
            boost_root.join("Library/include"),
            PathBuf::from("/opt/homebrew/opt/boost/include"),
        ],
        "boost/version.hpp",
    );

    cxx_build::bridge("src/chem.rs")
        .file("cpp/chem.cpp")
        .include("cpp")
        .include(include)
        .include(boost_include)
        .std("c++20")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-deprecated-copy")
        .compile("retrocast-rdkit");

    println!("cargo:rustc-link-search=native={}", lib.display());
    for name in [
        "RDKitSmilesParse",
        "RDKitRDInchiLib",
        "RDKitDescriptors",
        "RDKitGraphMol",
        "RDKitRDGeneral",
    ] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rerun-if-env-changed=RDKIT_ROOT");
    println!("cargo:rerun-if-env-changed=RDKIT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=RDKIT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BOOST_ROOT");
    println!("cargo:rerun-if-changed=cpp/chem.cpp");
    println!("cargo:rerun-if-changed=cpp/chem.hpp");
}

fn first_existing(candidates: &[PathBuf]) -> PathBuf {
    candidates
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or_else(|| candidates[0].clone())
}

fn first_containing(candidates: &[PathBuf], child: &str) -> PathBuf {
    candidates
        .iter()
        .find(|path| path.join(child).exists())
        .cloned()
        .unwrap_or_else(|| candidates[0].clone())
}

fn default_rdkit_root() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/opt/rdkit",
        "/usr/local/opt/rdkit",
        "/opt/rdkit",
    ] {
        let path = PathBuf::from(candidate);
        if path.join("include/rdkit/GraphMol/ROMol.h").exists() {
            return Some(path);
        }
    }
    None
}
