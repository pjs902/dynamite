#!/bin/bash
# One Slurm array task = one weight solve (spec section 5.3).
# $1 = manifest of model dirs, one per array task; this task takes the line
# at SLURM_ARRAY_TASK_ID.
set -euo pipefail
. "$(dirname "$0")/_select_item.sh"
CONFIG=${VERA_CONFIG:?VERA_CONFIG env var must point at the campaign yaml}
RUNDIR=${VERA_RUN_DIR:?VERA_RUN_DIR must be the driver run directory}
PY=${VERA_PYTHON:-python}

cd "$RUNDIR"
exec "$PY" -m dynamite.vera.solve_one --config "$CONFIG" --model-dir "$ITEM"
