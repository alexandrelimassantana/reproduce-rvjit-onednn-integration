#!/bin/bash
# Times batches the way oneDNN's performance CI does
# (.github/automation/performance/run_benchdnn_compare.sh): REPS repetitions
# of each batch, alternating between the two flavors, each a fresh benchdnn
# process printing one `%prb%,%-time%,%-ctime%` line per problem with
# --max-ms-per-prb=300. analysis/perfci_report.py then applies the CI's
# t-test to the samples. The warm-up batch of the driver runs first in every
# process.
#
# Usage: perfci.sh [options] batch... [-- ARGS]
#
#   batch           inputs/<batch>, e.g. performance-onednn/matmul.nightly
#   --dt LIST       comma-separated data types (default harness: the batch
#                   sets its own)
#   --id ID         names the logs (default: from the batch)
#   --flavors LIST  the baseline and the candidate (default upstream,rvjit)
#   --reps N        repetitions (default $REPS, else 10; the CI uses 50, its
#                   t-test needs 3)
#   -- ARGS         extra benchdnn arguments
#
# OMP_NUM_THREADS keeps its value if set, else 8. Logs are
# results/<machine>/<flavor>/perfci/<id>.<driver>.<dt>.t<threads>.csv.
#
# Examples:
#   ./perfci.sh performance-onednn/matmul.nightly performance-onednn/conv.nightly
#   ./perfci.sh --reps 50 --id onednn-ip-models performance-onednn/ip.models

set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dts=harness
id=
flavor_list=
reps=${REPS:-10}
batches=()
extra=()
while [ $# -gt 0 ]; do
    case $1 in
        --dt) dts=$2; shift 2 ;;
        --id) id=$2; shift 2 ;;
        --flavors) flavor_list=$2; shift 2 ;;
        --reps) reps=$2; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown option $1" ;;
        *) batches+=("$1"); shift ;;
    esac
done
[ ${#batches[@]} -ge 1 ] || { usage >&2; exit 2; }
require_riscv
flavors=($(flavors_of "$flavor_list")) || exit 2
[ ${#flavors[@]} -eq 2 ] || die "perfci compares two flavors, got ${flavors[*]}"
for dt in ${dts//,/ }; do dt_arg "$dt" > /dev/null; done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

for flavor in "${flavors[@]}"; do
    prefix=$INSTALL_DIR/$flavor
    [ -x "$prefix/tests/benchdnn/benchdnn" ] || die "no benchdnn in $prefix"
    mkdir -p "$(logs_of "$flavor" perfci)"
    write_run_env "$flavor" "$prefix" "$(logs_of "$flavor" perfci)"
    print_header "$flavor" "$prefix"
done

rc=0
for batch in "${batches[@]}"; do
    read -r driver harness <<< "$(resolve_batch "$batch")" || exit 2
    bid=${id:-$(echo "$batch" | tr '/.' '__')}
    for dt in ${dts//,/ }; do
        name=$(log_name "$bid" "$driver" "$dt")
        for flavor in "${flavors[@]}"; do
            : > "$(logs_of "$flavor" perfci)/$name.csv"
        done
        phase_begin perfci "${flavors[*]}" "$bid $batch $dt x$reps"
        for rep in $(seq 1 "$reps"); do
            for flavor in "${flavors[@]}"; do
                prefix=$INSTALL_DIR/$flavor
                LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                "$prefix/tests/benchdnn/benchdnn" --$driver --mode=P \
                        --max-ms-per-prb=300 \
                        "--perf-template=%prb%,%impl%,%-Gflops%,%-time%,%-ctime%" "${extra[@]}" \
                        --reset $(dt_arg "$dt") \
                        --batch="$INPUTS_DIR/common/warmup.$driver" \
                        --reset $(dt_arg "$dt") --batch="$harness" \
                        >> "$(logs_of "$flavor" perfci)/$name.csv" 2>&1 || rc=1
            done
        done
        phase_end $rc
        printf '%-8s %-26s %-10s t%-2s %d reps\n' "$bid" "$batch" "$dt" \
                "$OMP_NUM_THREADS" "$reps"
    done
done
phase_report
exit $rc
