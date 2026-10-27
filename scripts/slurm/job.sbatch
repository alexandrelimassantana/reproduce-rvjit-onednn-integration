#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#
# One slurm job: runs scripts/<script>.sh with the given arguments from the
# repository root. Submitted by submit.sh.
#
# Usage: sbatch [...] job.sbatch <script> ARGS...

set -uo pipefail

script=$1
shift
cd "${SLURM_SUBMIT_DIR:-.}"
echo "=== $(date -Iseconds) $script $* on $(hostname), job ${SLURM_JOB_ID:--}"
scripts/$script.sh "$@"
rc=$?
echo "=== $(date -Iseconds) done, exit status $rc"
exit $rc
