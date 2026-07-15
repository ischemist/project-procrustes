#!/usr/bin/env bash

set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must point at the checked-out repository}"

source_root="$GITHUB_WORKSPACE/.wheel-rdkit-src"
build_root="$GITHUB_WORKSPACE/.wheel-rdkit-build"
native_root="$GITHUB_WORKSPACE/.wheel-native"

dnf install -y cmake git ninja-build

cmake -S "$source_root" -B "$build_root" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$native_root" \
  -DRDK_BUILD_PYTHON_WRAPPERS=OFF \
  -DRDK_BUILD_CPP_TESTS=OFF \
  -DRDK_BUILD_INCHI_SUPPORT=ON \
  -DRDK_BUILD_DESCRIPTORS3D=OFF \
  -DRDK_BUILD_FREETYPE_SUPPORT=OFF \
  -DRDK_INSTALL_COMIC_FONTS=OFF \
  -DRDK_BUILD_CHEMDRAW_SUPPORT=OFF \
  -DRDK_BUILD_COORDGEN_SUPPORT=OFF \
  -DRDK_BUILD_MAEPARSER_SUPPORT=OFF \
  -DRDK_BUILD_MOLINTERCHANGE_SUPPORT=OFF \
  -DRDK_BUILD_PUBCHEMSHAPE_SUPPORT=OFF \
  -DRDK_BUILD_SLN_SUPPORT=OFF \
  -DRDK_INSTALL_INTREE=OFF \
  -DRDK_INSTALL_STATIC_LIBS=OFF \
  -DRDK_USE_BOOST_IOSTREAMS=OFF \
  -DRDK_USE_BOOST_SERIALIZATION=OFF \
  -DRDK_USE_BOOST_STACKTRACE=OFF \
  -DRDK_USE_URF=OFF

cmake --build "$build_root" --parallel "$(nproc)" --target Descriptors RDInchiLib

mkdir -p "$native_root/include/GraphMol" "$native_root/lib"
# The source tree keeps this public header with the vendored InChI API, while
# RDKit development packages install it under GraphMol/.
cp "$source_root/External/INCHI-API/inchi.h" "$native_root/include/GraphMol/inchi.h"
cp -a "$build_root/lib"/libRDKit*.so* "$native_root/lib/"

for library in SmilesParse RDInchiLib Descriptors GraphMol RDGeneral; do
  test -e "$native_root/lib/libRDKit${library}.so"
done
