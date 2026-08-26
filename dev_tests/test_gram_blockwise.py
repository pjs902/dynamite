"""gram_blockwise: P and q accumulated one row block at a time, so the design
matrix A never has to exist all at once.

The cvxopt/admm weight-solver branches only ever consume P = An^T An and
q = -An^T bn (An = A[1:]/col_norm, the total-mass row dropped as an equality
constraint - see test_cvxopt_equality.py). A itself is 371212 x 45000 = 124
GiB at float64 for omega Cen, and the reduction to P currently allocates a
second full-size copy on the way down, which is what pushes a production
solve to 250-430 GB peak. P is only ~15.1 GiB regardless of row count, so
accumulating it block-by-block (NormalEquationAccumulator, weight_solvers.py)
and discarding each block is the fix.

The two paths (materialize-then-reduce vs accumulate-blockwise) are
ALGEBRAICALLY IDENTICAL, not merely close - dsyrk's rank-k update and a
single A^T A are the same sum of outer products in a different order.
Anything above float64 rounding here is a bug, not noise.

THE ONE FAILURE MODE THAT MATTERS: construct_nnls_matrix_and_rhs and
construct_adelie_matrix_and_rhs both apply econ to the WHOLE matrix at once,
AFTER their assembly loop, because they hold the whole matrix. A blockwise
accumulator cannot do that: once a block is folded into G by dsyrk it can
never be rescaled row-by-row again. Getting this backwards - accumulating
raw blocks and dividing G by an aggregate "econ" afterwards - does not
raise; it silently produces a plausible but wrong P. test_econ_before_not_
after_accumulation pins this by constructing that exact wrong path and
showing it disagrees with the correct one well above rounding.

Run from the repo root:
    PYTHONPATH=. python dev_tests/test_gram_blockwise.py
"""

import os
import resource
import subprocess
import sys
import textwrap

import numpy as np

from dynamite.weight_solvers import NormalEquationAccumulator


def _matrix(seed=0, n=6000, p=250, blk=700, decades=14):
    """Column norms spanning many decades, like the real mass+kinematic rows
    (intrinsic masses ~1e-17, kinematic rows ~O(1))."""
    rng = np.random.default_rng(seed)
    A = np.abs(rng.standard_normal((n, p))) * 10.0 ** rng.uniform(
        -decades / 2, decades / 2, p
    )
    b = np.abs(rng.random(n)) * 10.0
    return A, b, blk


def _materialized_P_q(A, b):
    col_norm = np.linalg.norm(A, axis=0)
    col_norm[col_norm == 0] = 1.0
    b_max = np.abs(b).max()
    An, bn = A / col_norm, b / b_max
    P = An.T @ An
    q = -An.T @ bn
    return P, q, col_norm, b_max


def _accumulate(A, b, blk, dtype=np.float64):
    acc = NormalEquationAccumulator(A.shape[1], dtype=dtype)
    for i0 in range(0, A.shape[0], blk):
        acc.add(A[i0 : i0 + blk], b[i0 : i0 + blk])
    return acc


def test_blockwise_P_matches_materialized():
    A, b, blk = _matrix()
    P_ref, _, _, _ = _materialized_P_q(A, b)
    acc = _accumulate(A, b, blk)
    P, _, _, _ = acc.finalize()
    rel = np.abs(P - P_ref).max() / np.abs(P_ref).max()
    assert rel < 1e-12, rel


def test_blockwise_q_matches_materialized():
    A, b, blk = _matrix()
    _, q_ref, _, _ = _materialized_P_q(A, b)
    acc = _accumulate(A, b, blk)
    _, q, _, _ = acc.finalize()
    rel = np.abs(q - q_ref).max() / np.abs(q_ref).max()
    assert rel < 1e-12, rel


def test_col_norm_survives_the_column_spread():
    """col_norm falls out of sqrt(diag(G)) with nothing subtracted - that is
    exactly why it does not lose precision the way a two-pass mean/variance
    computation would across a 14-decade column spread."""
    A, b, blk = _matrix()
    col_ref = np.linalg.norm(A, axis=0)
    acc = _accumulate(A, b, blk)
    _, _, col, _ = acc.finalize()
    rel = np.abs(col - col_ref) / col_ref
    assert rel.max() < 1e-12, rel.max()


def test_different_block_sizes_agree():
    """The accumulation must not depend on how the rows happen to be
    chunked - blocks are an implementation detail of the streaming loop,
    not part of the math."""
    A, b, _ = _matrix()
    P1, q1, _, _ = _accumulate(A, b, 137).finalize()
    P2, q2, _, _ = _accumulate(A, b, 4000).finalize()
    assert np.abs(P1 - P2).max() / np.abs(P1).max() < 1e-13
    assert np.abs(q1 - q2).max() / np.abs(q1).max() < 1e-13


def test_econ_before_not_after_accumulation():
    """THE ordering trap. econ must divide each block BEFORE it is folded
    into G by dsyrk - not applied to G/v afterwards. This test builds both
    the correct path and the deliberately-wrong "divide after" path and
    shows they disagree well above rounding, so a future refactor that gets
    this backwards fails loudly instead of silently producing a plausible
    but wrong P.
    """
    rng = np.random.default_rng(3)
    n, p, blk = 4000, 150, 500
    A_raw = np.abs(rng.standard_normal((n, p))) * 10.0 ** rng.uniform(-6, 6, p)
    b_raw = np.abs(rng.random(n)) * 10.0
    # per-ROW errors that vary a lot from block to block (mimics intrinsic
    # mass rows at ~1e-17 next to kinematic rows at ~O(0.1))
    econ = 10.0 ** rng.uniform(-8, 2, n)
    A_correct = A_raw / econ[:, None]
    b_correct = b_raw / econ
    P_ref, q_ref, _, _ = _materialized_P_q(A_correct, b_correct)

    # CORRECT: divide each block by its own econ slice before acc.add()
    acc_correct = NormalEquationAccumulator(p)
    for i0 in range(0, n, blk):
        sl = slice(i0, i0 + blk)
        acc_correct.add(A_raw[sl] / econ[sl, None], b_raw[sl] / econ[sl])
    P_correct, q_correct, _, _ = acc_correct.finalize()
    assert np.abs(P_correct - P_ref).max() / np.abs(P_ref).max() < 1e-12
    assert np.abs(q_correct - q_ref).max() / np.abs(q_ref).max() < 1e-12

    # WRONG: accumulate raw blocks, then try to rescale G/v as if by a
    # single aggregate econ (e.g. its per-block mean) - this is the mistake
    # the ordering trap warns against. It cannot be corrected after the
    # fact because the per-row information was already summed away.
    acc_wrong = NormalEquationAccumulator(p)
    for i0 in range(0, n, blk):
        sl = slice(i0, i0 + blk)
        acc_wrong.add(A_raw[sl], b_raw[sl])
    econ_bar = econ.mean()  # the only kind of "after the fact" rescale
    # possible once rows are summed together - a single scalar
    G_wrong = acc_wrong.G / econ_bar ** 2
    v_wrong = acc_wrong.v / econ_bar
    col_wrong = np.sqrt(np.abs(np.diag(G_wrong)))
    col_wrong[col_wrong == 0] = 1.0
    b_max_wrong = np.abs(b_raw / econ_bar).max()
    P_wrong = G_wrong / np.outer(col_wrong, col_wrong)
    rel_wrong = np.abs(P_wrong - P_ref).max() / np.abs(P_ref).max()
    assert rel_wrong > 1e-3, (
        "the deliberately-wrong divide-after-accumulation path agreed with "
        f"the correct one to {rel_wrong:.3e} - this test no longer "
        "distinguishes the two, and needs a construction that does"
    )


def test_chi2_matches_residual():
    """chi2 = w'Gw - 2 w'v + ||b||^2 (the raw, pre-column-scaling
    accumulator outputs) must equal ||Aw - b||^2 computed directly."""
    A, b, blk = _matrix(n=3000, p=180)
    rng = np.random.default_rng(7)
    w = np.abs(rng.random(A.shape[1]))
    chi2_direct = float(np.sum((A @ w - b) ** 2))
    acc = _accumulate(A, b, blk)
    acc.finalize()  # mirrors G to full symmetric in place
    chi2_gram = float(w @ (acc.G @ w)) - 2.0 * float(w @ acc.v) + acc.b_sq_sum
    rel = abs(chi2_gram - chi2_direct) / abs(chi2_direct)
    assert rel < 1e-10, (chi2_gram, chi2_direct, rel)


def test_zero_column_left_unscaled():
    """A null orbit (all-zero column) must not divide by zero."""
    A, b, blk = _matrix(n=2000, p=50)
    A[:, 3] = 0.0
    acc = _accumulate(A, b, blk)
    P, q, col, _ = acc.finalize()
    assert col[3] == 1.0
    assert np.all(np.isfinite(P))
    assert np.all(np.isfinite(q))
    assert np.all(P[:, 3] == 0.0) and np.all(P[3, :] == 0.0)


_MEMORY_SUBPROCESS = textwrap.dedent(
    """
    import numpy as np, resource, sys
    sys.path.insert(0, {dynamite_dir!r})
    from dynamite.weight_solvers import NormalEquationAccumulator

    mode = sys.argv[1]
    n, p, blk = 900_000, 1800, 30_000
    rng = np.random.default_rng(0)

    if mode == "materialized":
        A = np.abs(rng.standard_normal((n, p))).astype(np.float64)
        b = np.abs(rng.random(n))
        col_norm = np.linalg.norm(A, axis=0)
        col_norm[col_norm == 0] = 1.0
        An = A / col_norm
        P = An.T @ An
    else:
        acc = NormalEquationAccumulator(p)
        for i0 in range(0, n, blk):
            block = np.abs(rng.standard_normal((blk, p)))
            acc.add(block, np.abs(rng.random(blk)))
        P, q, col, bm = acc.finalize()

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(peak_kb)
    """
)


def test_peak_memory_is_materially_lower_blockwise():
    """VmHWM (peak resident set), not current RSS: the materialized path's
    peak happens transiently while building A and reducing it to P, which a
    live-RSS snapshot at the end would miss entirely - it is exactly this
    transient peak (measured 250-430 GB in production) that blockwise
    avoids. Run each variant in its own subprocess so one peak cannot leak
    into the other's measurement.

    n=900000, p=1800 keeps A itself under ~13 GiB (float64) so this runs in
    a few seconds and stays well under the ~2 GB budget for the OTHER tests
    in this module - this is the one test allowed to be bigger.
    """
    dynamite_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = _MEMORY_SUBPROCESS.format(dynamite_dir=dynamite_dir)
    peaks = {}
    for mode in ("materialized", "blockwise"):
        out = subprocess.run(
            [sys.executable, "-c", script, mode],
            capture_output=True, text=True, check=True,
            env={**os.environ, "OMP_NUM_THREADS": "8"},
        )
        peaks[mode] = int(out.stdout.strip().splitlines()[-1])
    ratio = peaks["materialized"] / peaks["blockwise"]
    print(
        f"    peak RSS: materialized {peaks['materialized']/1e6:.2f} GB, "
        f"blockwise {peaks['blockwise']/1e6:.2f} GB, ratio {ratio:.2f}x"
    )
    assert ratio > 1.5, (
        f"expected blockwise peak RSS to be materially lower; got ratio "
        f"{ratio:.2f}x ({peaks})"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("gram_blockwise, OK")
