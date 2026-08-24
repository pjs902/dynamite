"""Single-library integration array task (spec section 5.2).

One process per chunk-family pair runs here, packed 12-wide per node by
integrate_package.sh. All orbit-integration machinery lives in
LegacyOrbitLibrary.get_orbit_library(); this wrapper only locates the row
and enforces cwd discipline.
"""

import sys

from .task_model import task_main


def _integrate(mod):
    mod.get_orblib()  # sentinel short-circuit applies here
    return {"integrated": True}


def main(argv=None):
    return task_main("one-library orbit integration", _integrate, argv)


if __name__ == "__main__":
    sys.exit(main())
