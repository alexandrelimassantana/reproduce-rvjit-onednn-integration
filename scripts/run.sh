#!/bin/bash
# Runs the evaluation on the flavors upstream and rvjit, natively on this RV64
# machine. It has five tasks, run (and submitted) in this order, performance
# first:
#
#   performance-onednn    oneDNN's perf-CI method over oneDNN's model shapes
#   performance-extended  benchdnn --mode=P over the IPDPS'26 convolutions
#                         and matmuls
#   ctest                 oneDNN's API gtests, one by one (scripts/ctest.sh)
#   correctness-onednn    benchdnn --mode=C over oneDNN's CI batches and shapes
#   correctness-extended  benchdnn --mode=C smoke tests of the PR's edges
#
# A task runs every batch of inputs/<task>/ whose file name starts with its
# benchdnn driver (matmul.*, ip.*, conv.*), one per primitive or dataset; ctest
# runs its tests (ctest.sh --list). A batch sets its own data types, or sweeps
# the ones of a `# dt: LIST` line.
#
# After each test or batch, scripts/report.py (Python 3, no package needed)
# writes its report to results/<machine>/reports/<task>.<batch>.md.
#
# Usage: run.sh [options] [TASK...]
#
#   TASK            one or more tasks (default all five)
#   --only NAMES    only these batches or ctest tests, e.g. conv.ci,ip.models
#   --flavors LIST  baseline and candidate (default upstream,rvjit)
#   --list          print the selected "<task> <batch or test>" pairs and stop
#
# THREADS (default 8) is the thread count. REPS sets the repetitions of
# performance-onednn (default 10). The exit status is nonzero when a job
# fails.
#
# Examples:
#   ./run.sh
#   ./run.sh correctness-onednn correctness-extended
#   ./run.sh performance-onednn --only ip.models
#   THREADS=4 REPS=50 ./run.sh performance-onednn

set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TASKS=(performance-onednn performance-extended ctest correctness-onednn
        correctness-extended)
tasks=()
only=
flavor_list=upstream,rvjit
list=0
while [ $# -gt 0 ]; do
    case $1 in
        --only) only=$2; shift 2 ;;
        --flavors) flavor_list=$2; shift 2 ;;
        --list) list=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; die "unknown argument $1" ;;
        *) case " ${TASKS[*]} " in
               *" $1 "*) tasks+=("$1"); shift ;;
               *) usage >&2; die "unknown task $1" ;;
           esac ;;
    esac
done
[ ${#tasks[@]} -gt 0 ] || tasks=("${TASKS[@]}")

# The "<task> <batch>" pairs selected by the tasks and --only
selected() {
    local task name
    for task in "${TASKS[@]}"; do
        case " ${tasks[*]} " in *" $task "*) ;; *) continue ;; esac
        if [ "$task" = ctest ]; then
            names=$("$scripts_dir/ctest.sh" --list)
        else
            names=$(cd "$INPUTS_DIR/$task" && ls -1 matmul.* ip.* conv.* 2>/dev/null)
        fi
        for name in $names; do
            [ -z "$only" ] || case ,$only, in *,$name,*) ;; *) continue ;; esac
            echo "$task $name"
        done
    done
}

if [ $list = 1 ]; then
    selected
    exit 0
fi
require_riscv
flavors_of "$flavor_list" > /dev/null || exit 2
export OMP_NUM_THREADS=${THREADS:-8}
export RUN_ID  # the job scripts' phases count towards this run's report

# Writes the report of one test or batch
report() {
    if command -v python3 > /dev/null; then
        python3 "$scripts_dir/report.py" --method "$1" --id "$2" --name "$3" \
                --base "${flavor_list%%,*}" --cand "${flavor_list##*,}" || rc=1
    else
        echo "no python3: no report for $3 (analysis/ covers it)"
    fi
}

rc=0
ran=0
phase_begin run "$flavor_list" "tasks=${tasks[*]} only=${only:-all}"
while read -r task name; do
    ran=$((ran + 1))
    common=(--flavors "$flavor_list")
    if [ "$task" = ctest ]; then
        echo
        echo "##### ctest: $name ($OMP_NUM_THREADS threads)"
        "$scripts_dir/ctest.sh" "${common[@]}" "^$name\$" || rc=1
        echo
        report ctest "$name" "ctest.$name"
        continue
    fi
    batch=$task/$name
    # The id names the logs: onednn- or extended-, then the batch
    id=${task#*-}-${name//./-}
    dts=$(sed -n 's/^#[[:space:]]*dt:[[:space:]]*//p' "$INPUTS_DIR/$batch" | head -1)
    dts=${dts:-harness}
    echo
    echo "##### $task: $name ($dts, $OMP_NUM_THREADS threads)"
    case $task in
        correctness-*) job=correctness ;;
        performance-onednn) job=perfci ;;
        performance-extended) job=performance ;;
    esac
    "$scripts_dir/$job.sh" "${common[@]}" --id "$id" --dt "$dts" "$batch" || rc=1
    echo
    report "${job/performance/perf}" "$id" "$task.$name"
done < <(selected)
phase_end $rc
[ $ran -gt 0 ] || die "no batch matches ${tasks[*]} --only ${only:-all}"
echo
phase_report
echo "exit status $rc; results in $RESULTS_DIR/$(machine_name)"
exit $rc
