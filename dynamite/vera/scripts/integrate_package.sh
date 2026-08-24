#!/bin/bash
# One Slurm array task = one exclusive p.vera node integrating a package of
# orbit libraries. $1 = manifest of ';'-joined model-dir packages, one per
# array task; this task takes the line at SLURM_ARRAY_TASK_ID.
# Libraries integrate exactly as LegacyOrbitLibrary expects: cwd = run dir,
# one thread per chunk-family process, GNU parallel packs the node.
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

IFS=';' read -ra MODELS <<< "$ITEM"
CONFIG=${VERA_CONFIG:?VERA_CONFIG env var must point at the campaign yaml}
RUNDIR=${VERA_RUN_DIR:?VERA_RUN_DIR must be the driver run directory}
PY=${VERA_PYTHON:-python}

cd "$RUNDIR"
printf '%s\n' "${MODELS[@]}" | xargs -P "${VERA_INT_PARALLEL:-12}" -n 1 -I{} \
    "$PY" -m dynamite.vera.integrate_one --config "$CONFIG" --model-dir {}
