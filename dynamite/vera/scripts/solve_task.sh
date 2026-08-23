#!/bin/bash
# One Slurm array task = one weight solve (spec section 5.3).
# $1 = model dir (from the driver's sbatch call).
set -euo pipefail
CONFIG=${VERA_CONFIG:?VERA_CONFIG env var must point at the campaign yaml}
RUNDIR=${VERA_RUN_DIR:?VERA_RUN_DIR must be the driver run directory}
PY=${VERA_PYTHON:-python}

cd "$RUNDIR"
exec "$PY" -m dynamite.vera.solve_one --config "$CONFIG" --model-dir "$1"
