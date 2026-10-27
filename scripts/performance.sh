#!/bin/bash
# Times batches: benchdnn --mode=P on each flavor, for each data type, one CSV
# per batch and data type. The warm-up batch of the driver
# (inputs/common/warmup.<driver>) runs first in every benchdnn process.
#
# Usage: performance.sh [options] batch... [-- ARGS]
#
#   batch           inputs/<batch>, e.g. brgemm/matmul.perf
#   --dt LIST       comma-separated data types (default f32), or `harness`
#   --id ID         names the logs (default: from the batch)
#   --flavors LIST  comma-separated flavors (default upstream,rvjit)
#   --times N       runs per problem (default 25; 0 lets benchdnn decide)
#   -- ARGS         extra benchdnn arguments, e.g. -- --max-ms-per-prb=2000
#
# OMP_NUM_THREADS keeps its value if set, else 8. Logs are
# results/<machine>/<flavor>/perf/<id>.<driver>.<dt>.t<threads>.csv beside a
# run.env; analysis/collect.py merges them.
#
# Examples:
#   ./performance.sh --dt f32,f16:f16:f32 performance-extended/conv.ipdps26
#   ./performance.sh --id extended-conv-ipdps26 --dt f32 performance-extended/conv.ipdps26

set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dts=f32
id=
flavor_list=
times=25
batches=()
extra=()
while [ $# -gt 0 ]; do
    case $1 in
        --dt) dts=$2; shift 2 ;;
        --id) id=$2; shift 2 ;;
        --flavors) flavor_list=$2; shift 2 ;;
        --times) times=$2; shift 2 ;;
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
[ "$times" -gt 0 ] && extra=(--fix-times-per-prb="$times" "${extra[@]}")

rc=0
for flavor in $flavors; do
    prefix=$INSTALL_DIR/$flavor
    benchdnn=$prefix/tests/benchdnn/benchdnn
    [ -x "$benchdnn" ] || die "no benchdnn in $prefix"
    logs=$(logs_of "$flavor" perf)
    mkdir -p "$logs"
    write_run_env "$flavor" "$prefix" "$logs"
    print_header "$flavor" "$prefix"
    export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    for batch in "${batches[@]}"; do
        read -r driver harness <<< "$(resolve_batch "$batch")" || exit 2
        bid=${id:-$(echo "$batch" | tr '/.' '__')}
        for dt in ${dts//,/ }; do
            log=$logs/$(log_name "$bid" "$driver" "$dt").csv
            phase_begin performance "$flavor" "$bid $batch $dt"
            "$benchdnn" --$driver --mode=P --perf-template=csv "${extra[@]}" \
                    --reset $(dt_arg "$dt") \
                    --batch="$INPUTS_DIR/common/warmup.$driver" \
                    --reset $(dt_arg "$dt") --batch="$harness" > "$log" 2>&1
            status=$?
            phase_end $status
            [ $status -ne 0 ] && rc=1
            printf '%-8s %-26s %-16s t%-2s %5s problems, rc=%d\n' "$bid" \
                    "$batch" "$dt" "$OMP_NUM_THREADS" \
                    "$(grep -c '^perf,cpu' "$log")" $status
        done
    done
    echo "logs in $logs"
done
phase_report
exit $rc
