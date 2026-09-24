"""check_blas_threading must refuse multi-threaded OpenBLAS and allow the rest.

Runs the check in a fresh interpreter per case, since thread counts are read
from the environment at BLAS load time.
"""
import os
import subprocess
import sys

import pytest

CODE = ("import logging; from dynamite.model_iterator import check_blas_threading;"
        "check_blas_threading(logging.getLogger())")


def run(**env):
    e = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
         'MKL_NUM_THREADS': '1', **env}
    return subprocess.run([sys.executable, '-c', CODE], env=e,
                          capture_output=True, text=True).returncode


def uses_openblas():
    import scipy.linalg  # noqa: F401
    from threadpoolctl import threadpool_info
    return any(i['internal_api'] == 'openblas' and 'cvxopt' not in i['filepath']
               for i in threadpool_info() if i['user_api'] == 'blas')


def test_single_threaded_passes():
    assert run() == 0


@pytest.mark.skipif(not uses_openblas(), reason='needs an OpenBLAS-backed scipy')
def test_threaded_openblas_refused():
    assert run(OPENBLAS_NUM_THREADS='8') != 0


@pytest.mark.skipif(uses_openblas(), reason='needs an MKL-backed scipy')
def test_threaded_mkl_passes():
    assert run(MKL_NUM_THREADS='8') == 0
