#!/usr/bin/env bash
# Build the MoGe3 OFX plugin on Linux (gcc/clang + cmake; ninja or make).
#   ofx/build.sh [Release|Debug]
# Output: ofx/build-linux/MoGe3.ofx.bundle/Contents/Linux-x86-64/MoGe3.ofx
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
config="${1:-Release}"
gen=()
command -v ninja >/dev/null 2>&1 && gen=(-G Ninja)
cmake -S "$here" -B "$here/build-linux" "${gen[@]}" -DCMAKE_BUILD_TYPE="$config"
cmake --build "$here/build-linux" --config "$config"
echo "built: $here/build-linux/MoGe3.ofx.bundle/Contents/Linux-x86-64/MoGe3.ofx"
