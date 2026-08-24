#!/bin/bash
# One Slurm array task = one weight solve (spec section 5.3).
# $1 = manifest of model dirs, one per array task; this task takes the line
# at SLURM_ARRAY_TASK_ID.
set -euo pipefail

# Inlined on purpose, NOT sourced from a sibling file: sbatch runs a COPY of
# this script from the node spool dir, so "$(dirname "$0")" is not the
# package directory and any `.` of a neighbouring file fails there.
MANIFEST="${1:?manifest path required}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ITEM=$(sed -n "$((IDX + 1))p" "$MANIFEST")
if [ -z "$ITEM" ]; then
    echo "no item at index $IDX in $MANIFEST" >&2
    exit 2
fi

CONFIG=${VERA_CONFIG:?VERA_CONFIG env var must point at the campaign yaml}
RUNDIR=${VERA_RUN_DIR:?VERA_RUN_DIR must be the driver run directory}
PY=${VERA_PYTHON:-python}

cd "$RUNDIR"
exec "$PY" -m dynamite.vera.solve_one --config "$CONFIG" --model-dir "$ITEM"
