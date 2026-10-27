#!/bin/bash
# Runs oneDNN's API gtests of matmul, inner product and convolution from each
# install. The benchdnn CI batches of ctest also run backward problems, which
# the inference builds lack: correctness-onednn runs their inference part
# (conv.test_ci, ip.test_ci) instead.
#
# Usage: ctest.sh [options] [regex]
#
#   regex           only the tests whose name matches (grep -E)
#   --flavors LIST  comma-separated flavors (default upstream,rvjit)
#   --list          print the names of the tests and stop
#
# The tests mirror the build tree's ctest entries:
#   test_matmul, test_gemm_f32, test_inner_product_forward,
#   test_convolution_{forward,eltwise_forward}_*
# OMP_NUM_THREADS keeps its value if set, else 8. Logs are
# results/<machine>/<flavor>/ctest/<test>.log, with the test's result in
# <test>.result; summary gathers the results of every test run so far, and the
# results of the flavors are compared at the end.
#
# Examples:
#   ./ctest.sh
#   ./ctest.sh 'matmul|gemm'

set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

flavor_list=
pos=()
list=0
while [ $# -gt 0 ]; do
    case $1 in
        --flavors) flavor_list=$2; shift 2 ;;
        --list) list=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown option $1" ;;
        *) pos+=("$1"); shift ;;
    esac
done
[ ${#pos[@]} -le 1 ] || { usage >&2; exit 2; }
filter=${pos[0]:-.}

tests=(
    "test_matmul|gtests|test_matmul"
    "test_gemm_f32|gtests|test_gemm_f32"
    "test_inner_product_forward|gtests|test_inner_product_forward"
    "test_convolution_forward_f32|gtests|test_convolution_forward_f32"
    "test_convolution_forward_u8s8fp|gtests|test_convolution_forward_u8s8fp"
    "test_convolution_forward_u8s8s32|gtests|test_convolution_forward_u8s8s32"
    "test_convolution_eltwise_forward_f32|gtests|test_convolution_eltwise_forward_f32"
    "test_convolution_eltwise_forward_x8s8f32s32|gtests|test_convolution_eltwise_forward_x8s8f32s32"
)

if [ $list = 1 ]; then
    for entry in "${tests[@]}"; do
        echo "${entry%%|*}"
    done | grep -E -- "$filter"
    exit 0
fi
require_riscv
flavors=$(flavors_of "$flavor_list") || exit 2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

rc=0
results=()
for flavor in $flavors; do
    prefix=$INSTALL_DIR/$flavor
    [ -d "$prefix/tests" ] || die "no tests in $prefix"
    logs=$(logs_of "$flavor" ctest)
    mkdir -p "$logs"
    results+=("$flavor")
    write_run_env "$flavor" "$prefix" "$logs"
    print_header "$flavor" "$prefix"
    export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    failed=0
    ran=0
    for entry in "${tests[@]}"; do
        IFS='|' read -r name dir cmd <<< "$entry"
        echo "$name" | grep -qE -- "$filter" || continue
        read -r -a args <<< "$cmd"
        args[0]="./${args[0]}"
        phase_begin ctest "$flavor" "$name"
        (cd "$prefix/tests/$dir" && "${args[@]}") > "$logs/$name.log" 2>&1
        status=$?
        phase_end $status
        ran=$((ran + 1))
        result=Passed
        if [ $status -ne 0 ]; then
            result="***Failed (rc=$status)"
            failed=$((failed + 1))
            rc=1
        fi
        printf '%-46s %s\n' "$name" "$result" | tee "$logs/$name.result"
    done
    # Concurrent jobs run one test each: gather every result so far
    cat "$logs"/*.result > "$logs/summary" 2> /dev/null
    echo "$((ran - failed))/$ran passed; logs in $logs"
done

# The results of this run's tests, per flavor
run_results() {
    local f name
    for name in $("$0" --list "$filter"); do
        f=$(logs_of "$1" ctest)/$name.result
        [ -f "$f" ] && cat "$f"
    done
}
if [ ${#results[@]} -gt 1 ]; then
    echo "=== ${results[0]} vs ${results[1]}"
    if diff <(run_results "${results[0]}") <(run_results "${results[1]}") > /dev/null; then
        echo "same results"
    else
        diff <(run_results "${results[0]}") <(run_results "${results[1]}")
    fi
fi
phase_report
exit $rc
