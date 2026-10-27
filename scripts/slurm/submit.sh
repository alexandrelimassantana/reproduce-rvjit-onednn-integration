#!/bin/bash
# Submits work to slurm, one exclusive node per job, from the repository root.
#
# Usage: submit.sh --partition P [--time T] run [TASK...] [run.sh options]
#        submit.sh --partition P [--time T] <script> ARGS...
#
#   --partition P  slurm partition (or SLURM_PARTITION); required
#   --time T       time limit of each job (or SLURM_TIME; default 03:00:00)
#   run            one independent job per ctest test, correctness batch (one
#                  per primitive) and performance dataset of the tasks
#                  (default all five) that run.sh selects; each job writes its
#                  report to results/<machine>/reports/<task>.<batch>.md
#   script ARGS    one job running another script of scripts/ (ctest,
#                  correctness, performance, perfci, build --native)
#
# THREADS (default 8) becomes each job's thread count; REPS passes through.
# DRY_RUN=1 prints the sbatch commands instead. The job logs are
# results/slurm/<task>.<batch or test, or script>-<job>.out.
#
# Examples:
#   scripts/slurm/submit.sh --partition P run
#   scripts/slurm/submit.sh --partition P run correctness-onednn
#   scripts/slurm/submit.sh --partition P --time 04:00:00 run performance-onednn --only conv.nightly

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
usage() { sed -n '/^# Usage/,/^# Examples/p' "$0" | sed 's/^# \{0,1\}//'; }

partition=${SLURM_PARTITION:-}
time=${SLURM_TIME:-03:00:00}
while [ $# -gt 0 ]; do
    case $1 in
        --partition) partition=$2; shift 2 ;;
        --time) time=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) break ;;
    esac
done
[ $# -ge 1 ] || { usage >&2; exit 2; }
[ -n "$partition" ] || { usage >&2; echo "submit.sh: no partition (--partition or SLURM_PARTITION)" >&2; exit 2; }
script=$1
shift
[ -x "$root/scripts/$script.sh" ] || { echo "submit.sh: no scripts/$script.sh" >&2; exit 2; }

threads=${THREADS:-8}

# Submits one job named $1 that runs scripts/$2.sh with the remaining arguments
submit() {
    local name=$1
    shift
    local cmd=(sbatch --partition="$partition" --time="$time" --chdir="$root"
            --job-name="rvjit-$name" --output="$root/results/slurm/$name-%j.out"
            --export="ALL,OMP_NUM_THREADS=$threads,THREADS=$threads"
            "$root/scripts/slurm/job.sbatch" "$@")
    if [ -n "${DRY_RUN:-}" ]; then
        printf '%q ' "${cmd[@]}"
        echo
    else
        "${cmd[@]}"
    fi
}

[ -n "${DRY_RUN:-}" ] || mkdir -p "$root/results/slurm"
if [ "$script" = run ]; then
    pairs=$("$root/scripts/run.sh" --list "$@")
    [ -n "$pairs" ] || { echo "submit.sh: no batch selected" >&2; exit 2; }
    # Each job names its task and batch; --flavors passes on
    args=()
    while [ $# -gt 0 ]; do
        case $1 in
            --flavors) [ $# -ge 2 ] || { usage >&2; exit 2; }; args+=("$1" "$2"); shift 2 ;;
            --only) shift 2 ;;
            *) shift ;;
        esac
    done
    while read -r task name; do
        submit "$task.$name" run "$task" --only "$name" "${args[@]}"
    done <<< "$pairs"
else
    submit "$script" "$script" "$@"
fi
