#!/bin/bash
# Builds the versions of versions.conf (fetching them first when missing) and
# packages each into install/<name>: the library, benchdnn with its matmul,
# ip and conv inputs, and the gtests of matmul, gemm, inner product and
# convolution. install/ is what the RV64 machine needs.
#
# Usage: build.sh [options] [name...|all]
#
#   --native          build with the host compilers (on an RV64 machine;
#                     the default there)
#   --toolchain FILE  cross-compile with this CMake toolchain file
#   --image IMG       cross-compile in this container image (default
#                     rvjit-repro:gcc14, built from container/Containerfile
#                     when missing)
#   -j N              parallel jobs (default: nproc)
#
# Without an option, CC and CXX (both set) select the local cross compilers
# (RISCV_SYSROOT optionally names their sysroot); otherwise the build runs in
# the container (CONTAINER=podman|docker, default: whichever is installed).
# Re-running reuses build/<name> incrementally. install/<name>/REVISION
# records the commit and the vendored rvjit (its release tag and commit),
# install/<name>/BUILD the
# toolchain, the library size and how long each step took.
#
# Examples:
#   ./build.sh
#   CC=riscv64-linux-gnu-gcc-14 CXX=riscv64-linux-gnu-g++-14 ./build.sh rvjit
#   ./build.sh --toolchain ~/riscv.cmake -j 16 all
#   ./build.sh --native             # on the RV64 machine

set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SOURCES_DIR=${SOURCES_DIR:-$root_dir/sources}
BUILD_DIR=${BUILD_DIR:-$root_dir/build}
IMAGE_DEFAULT=rvjit-repro:gcc14

mode=
toolchain=
image=
jobs=$(nproc)
names=()
while [ $# -gt 0 ]; do
    case $1 in
        --native) mode=native; shift ;;
        --toolchain) mode=toolchain; toolchain=$(readlink -f "$2"); shift 2 ;;
        --image) mode=container; image=$2; shift 2 ;;
        -j) jobs=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown option $1" ;;
        all) shift ;;
        *) names+=("$1"); shift ;;
    esac
done
if [ ${#names[@]} -eq 0 ]; then
    mapfile -t names < <(grep -Ev '^[[:space:]]*(#|$)' "$root_dir/versions.conf" \
            | awk '{ print $1 }')
fi
if [ -z "$mode" ]; then
    if [ "$(uname -m)" = riscv64 ]; then
        mode=native
    elif [ -n "${CC:-}" ] && [ -n "${CXX:-}" ]; then
        mode=local
    else
        mode=container
    fi
fi

gtests=(test_matmul test_gemm_f32 test_inner_product_forward
        test_convolution_forward_f32 test_convolution_forward_u8s8fp
        test_convolution_forward_u8s8s32 test_convolution_eltwise_forward_f32
        test_convolution_eltwise_forward_x8s8f32s32)

# One step of the build: configure, compile or package; runs where the
# compiler is (here, or in the container). $1 is the step, then the source,
# build and install trees, the jobs, the toolchain file ('' for native), then
# the gtests.
build_step() {
    local step=$1 src=$2 build=$3 install=$4 jobs=$5 toolchain=$6
    shift 6
    local gtests=("$@")
    case $step in
    configure)
        local opts=(-DCMAKE_BUILD_TYPE=Release -DDNNL_TARGET_ARCH=RV64
                -DDNNL_CPU_RUNTIME=OMP -DONEDNN_BUILD_GRAPH=OFF
                -DONEDNN_ENABLE_WORKLOAD=INFERENCE
                "-DONEDNN_ENABLE_PRIMITIVE=CONVOLUTION;MATMUL;INNER_PRODUCT;REORDER"
                -DONEDNN_BUILD_TESTS=ON -DONEDNN_BUILD_EXAMPLES=OFF
                -DONEDNN_BUILD_DOC=OFF)
        [ -z "$toolchain" ] || opts+=(-DCMAKE_TOOLCHAIN_FILE="$toolchain")
        cmake -S "$src" -B "$build" "${opts[@]}" > "$build.configure.log" 2>&1 \
                || { tail -40 "$build.configure.log"; return 1; }
        ;;
    compile)
        cmake --build "$build" -j "$jobs" --target dnnl benchdnn "${gtests[@]}" \
                > "$build.build.log" 2>&1 || { tail -40 "$build.build.log"; return 1; }
        ;;
    package)
        rm -rf "$install"
        cmake --install "$build" --prefix "$install" > /dev/null
        mkdir -p "$install/tests/benchdnn/inputs" "$install/tests/gtests"
        cp "$build/tests/benchdnn/benchdnn" "$install/tests/benchdnn/"
        cp -r "$src"/tests/benchdnn/inputs/{matmul,ip,conv} \
                "$install/tests/benchdnn/inputs/"
        local t
        for t in "${gtests[@]}"; do
            cp "$build/tests/gtests/$t" "$install/tests/gtests/"
        done
        ;;
    compiler)
        local cxx
        cxx=$(sed -n 's/^set(CMAKE_CXX_COMPILER "\(.*\)")$/\1/p' \
                "$build"/CMakeFiles/*/CMakeCXXCompiler.cmake | head -1)
        echo "$cxx: $("$cxx" --version | head -1)"
        ;;
    esac
}

# Runs one build step with the chosen toolchain
run_step() {
    case $mode in
    native)
        build_step "$@" ;;
    local|toolchain)
        build_step "${@:1:5}" "$toolchain" "${@:7}" ;;
    container)
        local args=(--rm -v "$root_dir:$root_dir" -w "$root_dir")
        [ "$engine" = docker ] && args+=(--user "$(id -u):$(id -g)")
        "$engine" run "${args[@]}" "$image" bash -c \
                "set -euo pipefail; $(declare -f build_step); build_step \"\$@\"" _ \
                "${@:1:5}" "$root_dir/container/riscv64-gcc14.cmake" "${@:7}" ;;
    esac
}

# The toolchain file of the local cross compilers in CC and CXX
write_local_toolchain() {
    local file=$BUILD_DIR/toolchain-local.cmake
    {
        echo "set(CMAKE_SYSTEM_NAME Linux)"
        echo "set(CMAKE_SYSTEM_PROCESSOR riscv64)"
        echo "set(CMAKE_C_COMPILER \"$(command -v "$CC")\")"
        echo "set(CMAKE_CXX_COMPILER \"$(command -v "$CXX")\")"
        if [ -n "${RISCV_SYSROOT:-}" ]; then
            echo "set(CMAKE_SYSROOT \"$RISCV_SYSROOT\")"
            echo "set(CMAKE_FIND_ROOT_PATH \"$RISCV_SYSROOT\")"
        fi
        echo "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)"
        echo "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)"
        echo "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)"
        echo "set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)"
    } > "$file"
    echo "$file"
}

mkdir -p "$BUILD_DIR"
image_id=local
case $mode in
local)
    toolchain=$(write_local_toolchain) ;;
container)
    image=${image:-$IMAGE_DEFAULT}
    engine=${CONTAINER:-$(command -v podman > /dev/null && echo podman || echo docker)}
    command -v "$engine" > /dev/null \
            || die "no podman or docker; set CC and CXX, or use --toolchain or --native"
    if ! "$engine" image inspect "$image" > /dev/null 2>&1; then
        [ "$image" = "$IMAGE_DEFAULT" ] || die "no image $image"
        phase_begin image "-" "$image"
        echo "=== building $image from container/Containerfile"
        "$engine" build -t "$image" -f "$root_dir/container/Containerfile" \
                "$root_dir/container"
        phase_end 0
    fi
    image_id="$image $("$engine" image inspect --format '{{.Id}}' "$image" | cut -c1-19)" ;;
esac
echo "=== toolchain: $mode${toolchain:+ ($toolchain)}${image:+ ($image)}"

for name in "${names[@]}"; do
    src=$SOURCES_DIR/$name
    [ -d "$src" ] || "$scripts_dir/fetch.sh" "$name"
    build=$BUILD_DIR/$name
    install=$INSTALL_DIR/$name
    echo "=== $name: $src -> $install"
    phase_begin build "$name" "$mode"
    timing=()
    for step in configure compile package; do
        start=$(date +%s)
        phase_begin "$step" "$name" "$mode"
        rc=0
        run_step $step "$src" "$build" "$install" "$jobs" "" "${gtests[@]}" || rc=$?
        phase_end $rc
        [ $rc = 0 ] || { phase_end $rc; die "$name: $step failed (logs in $build.*.log)"; }
        timing+=("$step=$(($(date +%s) - start))s")
        echo "    $step: $(($(date +%s) - start))s"
    done
    phase_end 0

    # The oneDNN commit (with -dirty for local changes) and the rvjit it vendors
    rev="$name: $(git -C "$src" log -1 --format='%h %s')"
    [ -n "$(git -C "$src" status --porcelain --untracked-files=no)" ] \
            && rev="$rev -dirty"
    vendored=$src/third_party/rvjit/README.md
    rvjit=
    if [ -f "$vendored" ]; then
        version=$(sed -n 's/^version: //p' "$vendored")
        commit=$(sed -n 's/^revision: //p' "$vendored")
        rvjit="${version:+$version }$commit"
        rev="$rev (rvjit ${version:+$version }${commit:0:12})"
    fi
    echo "$rev" > "$install/REVISION"
    {
        echo "name=$name"
        echo "commit=$(git -C "$src" rev-parse HEAD)"
        echo "rvjit=${rvjit:-none}"
        echo "toolchain=$mode${toolchain:+ $toolchain}"
        echo "image=$image_id"
        echo "compiler=$(run_step compiler "$src" "$build" "$install" "$jobs" "")"
        echo "libdnnl_bytes=$(stat -L -c %s "$install"/lib*/libdnnl.so)"
        echo "build_host=$(host_name)"
        echo "jobs=$jobs"
        echo "timing=${timing[*]}"
        echo "date=$(date -Iseconds)"
    } > "$install/BUILD"
    echo "$rev"
done
phase_report
