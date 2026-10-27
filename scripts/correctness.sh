#!/bin/bash
# Checks batches for correctness: benchdnn --mode=C on each flavor, for each
# data type. Each log ends with benchdnn's implementation statistics, which
# run.sh and the analysis read to tell which implementation ran.
#
# Usage: correctness.sh [options] batch... [-- ARGS]
#
#   batch           inputs/<batch>, e.g. brgemm/matmul.ci
#   --dt LIST       comma-separated data types (default f32), or `harness`
#   --id ID         names the logs (default: from the batch)
#   --flavors LIST  comma-separated flavors (default upstream,rvjit)
#   -- ARGS         extra benchdnn arguments
#
# OMP_NUM_THREADS keeps its value if set, else 8. Logs are
# results/<machine>/<flavor>/correctness/<id>.<driver>.<dt>.t<threads>.log,
# with one summary line each in summary. The exit status is nonzero when a
# problem fails or benchdnn crashes.
#
# Examples:
#   ./correctness.sh --dt harness correctness-onednn/ip.ci
#   ./correctness.sh --id onednn-matmul-ci --dt harness correctness-onednn/matmul.ci

set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dts=f32
id=
flavor_list=
batches=()
extra=()
while [ $# -gt 0 ]; do
    case $1 in
        --dt) dts=$2; shift 2 ;;
        --id) id=$2; shift 2 ;;
        --flavors) flavor_list=$2; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown option $1" ;;
        *) batches+=("$1"); shift ;;
    esac
done
[ ${#batches[@]} -ge 1 ] || { usage >&2; exit 2; }
require_riscv
flavors=$(flavors_of "$flavor_list") || exit 2
for dt in ${dts//,/ }; do dt_arg "$dt" > /dev/null; done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

rc=0
for flavor in $flavors; do
    prefix=$INSTALL_DIR/$flavor
    benchdnn=$prefix/tests/benchdnn/benchdnn
    [ -x "$benchdnn" ] || die "no benchdnn in $prefix"
    logs=$(logs_of "$flavor" correctness)
    mkdir -p "$logs"
    write_run_env "$flavor" "$prefix" "$logs"
    print_header "$flavor" "$prefix"
    export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    for batch in "${batches[@]}"; do
        read -r driver harness <<< "$(resolve_batch "$batch")" || exit 2
        bid=${id:-$(echo "$batch" | tr '/.' '__')}
        for dt in ${dts//,/ }; do
            log=$logs/$(log_name "$bid" "$driver" "$dt").log
            phase_begin correctness "$flavor" "$bid $batch $dt"
            "$benchdnn" --$driver --mode=C --reset $(dt_arg "$dt") \
                    "${extra[@]}" --batch="$harness" > "$log" 2>&1
            status=$?
            phase_end $status
            line=$(grep -E "^tests:" "$log" | tail -1)
            [ -n "$line" ] || line="crashed (rc=$status)"
            if [ $status -ne 0 ] || ! echo "$line" | grep -q "failed:0"; then
                rc=1
            fi
            printf '%-8s %-26s %-16s t%-2s %s\n' "$bid" "$batch" "$dt" \
                    "$OMP_NUM_THREADS" "$line" | tee -a "$logs/summary"
        done
    done
    echo "logs in $logs"
done
phase_report
exit $rc
