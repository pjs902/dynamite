# Resolve THIS array task's item from the manifest ($1).
# Slurm gives every task of an array identical argv, so the per-task argument
# must come from SLURM_ARRAY_TASK_ID, not from the command line.
MANIFEST="${1:?manifest path required}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
ITEM=$(sed -n "$((IDX + 1))p" "$MANIFEST")
if [ -z "$ITEM" ]; then
    echo "no item at index $IDX in $MANIFEST" >&2
    exit 2
fi
