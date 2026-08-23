"""Group libraries into node-sized work packages (spec section 5.2)."""

PROCS_PER_LIB_DEFAULT = 6  # orblib_chunks(3) x orbit families(2), 1 thread each
CORES_PER_VERA_NODE = 72


def pack_libraries(
    model_dirs, procs_per_lib=PROCS_PER_LIB_DEFAULT, cores=CORES_PER_VERA_NODE
):
    """Order-preserving grouping into packages of cores // procs_per_lib."""
    size = max(1, cores // procs_per_lib)
    return [model_dirs[i : i + size] for i in range(0, len(model_dirs), size)]
