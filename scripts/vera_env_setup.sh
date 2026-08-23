#!/bin/bash
# VERA environment bootstrap (spec section 7). Idempotent.
# Run on a VERA login node: bash scripts/vera_env_setup.sh
set -euo pipefail

BASE=/vera/ptmp/gc/mia/pesmith/oCen
ENV=$BASE/envs/dynamite
REPO=$(cd "$(dirname "$0")/.." && pwd)

# Toolchain for legacy_fortran. Pin the exact version after checking:
#   module av gcc
module purge
module load gcc

mkdir -p "$BASE"

if [ ! -x "$ENV/bin/python" ]; then
    conda create -y -p "$ENV" python=3.12
fi
"$ENV/bin/python" -m pip install --upgrade pip

# Numeric stack first (numpy pin decides wheel compatibility), then adelie,
# then dynamite itself. No galahad: the NNLS path is retired.
"$ENV/bin/python" -m pip install "numpy" scipy astropy pathos possum \
    matplotlib pandas pyyaml dill multiprocess
"$ENV/bin/python" -m pip install adelie
"$ENV/bin/python" -m pip install --no-deps "$REPO"
# Phase-3 BO stack (CPU torch). Large wheels; scratch env keeps /u quota safe.
"$ENV/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/python" -m pip install botorch gpytorch

make -C "$REPO/legacy_fortran" all

FREEZE="$BASE/ENV_FREEZE.txt"
{
    echo "# VERA environment freeze - $(date -u +%Y-%m-%dT%H:%MZ)"
    "$ENV/bin/python" - <<'EOF'
import sys
import numpy, scipy, astropy
print("python", sys.version.split()[0])
for m in (numpy, scipy, astropy):
    print(m.__name__, m.__version__)
try:
    import adelie
    print("adelie", adelie.__version__)
except Exception as e:
    print("adelie IMPORT FAILED:", repr(e))
try:
    import torch, botorch, gpytorch
    print("torch", torch.__version__, "| botorch", botorch.__version__,
          "| gpytorch", gpytorch.__version__)
except Exception as e:
    print("BO stack unavailable:", repr(e))
EOF
} > "$FREEZE"

"$ENV/bin/python" -c "import dynamite, adelie" \
    && echo "vera env OK -> $FREEZE" || { echo "env broken"; exit 1; }
