use std::env;

fn main() {
    // RDKit's SMILES/InChI pipeline can exceed the 1 MiB stack reserved by a
    // default Windows executable. Python hosts normally reserve more, while
    // the standalone CLI owns this setting and must make it explicit.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo:rustc-link-arg-bin=retrocast=/STACK:8388608");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
