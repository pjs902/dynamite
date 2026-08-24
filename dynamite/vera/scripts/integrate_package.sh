#!/bin/bash
# One Slurm array task = one exclusive p.vera node integrating a package of
# orbit libraries. $1 = manifest of ';'-joined model-dir packages, one per
# array task; this task takes the line at SLURM_ARRAY_TASK_ID.
# Libraries integrate exactly as LegacyOrbitLibrary expects: cwd = run dir,
# one thread per chunk-family process, GNU parallel packs the node.
set -euo pipefail
. "$(dirname "$0")/_select_item.sh"
IFS=';' read -ra MODELS <<< "$ITEM"
CONFIG=${VERA_CONFIG:?VERA_CONFIG env var must point at the campaign yaml}
RUNDIR=${VERA_RUN_DIR:?VERA_RUN_DIR must be the driver run directory}
PY=${VERA_PYTHON:-python}

cd "$RUNDIR"
printf '%s\n' "${MODELS[@]}" | xargs -P "${VERA_INT_PARALLEL:-12}" -n 1 -I{} \
    "$PY" -m dynamite.vera.integrate_one --config "$CONFIG" --model-dir {}
