"""Regression tests for the two low-level linear-algebra helpers on the
Gram/blockwise path: NormalEquationAccumulator.finalize()'s in-place
symmetric mirror + column scaling, and _matvec_f64's chunked gemv.

Both were rewritten to bound their temporaries (finalize no longer builds
np.triu_indices or np.outer; _matvec_f64 no longer lets NumPy promote a
whole float32 matrix to float64). Neither may change the numbers beyond
round-off, which is what these tests pin.

Run: PYTHONPATH=<repo> python test_matvec_and_mirror.py
"""
import numpy as np
import pytest
import tracemalloc

from dynamite.weight_solvers import NormalEquationAccumulator, _matvec_f64


def _rel(got, ref):
    """Max abs deviation as a fraction of the reference vector's own scale."""
    return np.abs(got - ref).max() / max(np.abs(ref).max(), np.finfo(float).tiny)


def test_finalize_matches_a_materialized_computation():
    """P, q, col_norm from the accumulator == the textbook formulas on A."""
    rng = np.random.default_rng(0)
    for n_orbs, block in ((7, 3), (129, 64), (4109, 1024)):
        NormalEquationAccumulator._MIRROR_BLOCK = block
        acc = NormalEquationAccumulator(n_orbs)
        A = rng.standard_normal((50, n_orbs))
        b = rng.standard_normal(50)
        acc.add(A, b)
        P, q, col, b_max = acc.finalize()
        G = A.T @ A
        col_ref = np.sqrt(np.abs(np.diag(G)))
        assert np.allclose(acc.G, G, rtol=1e-12, atol=1e-12), n_orbs
        assert np.allclose(col, col_ref, rtol=1e-13), n_orbs
        assert np.allclose(P, G / np.outer(col_ref, col_ref), rtol=1e-12), n_orbs
        assert np.allclose(q, -(A.T @ b) / (col_ref * b_max), rtol=1e-12), n_orbs
    print("test_finalize_matches_a_materialized_computation: OK")


def test_finalize_mirror_is_exactly_symmetric():
    """The block mirror must fill the lower triangle exactly, not approximately."""
    rng = np.random.default_rng(1)
    for n_orbs, block in ((5, 2), (100, 32), (257, 256), (256, 256)):
        NormalEquationAccumulator._MIRROR_BLOCK = block
        acc = NormalEquationAccumulator(n_orbs)
        acc.add(rng.standard_normal((20, n_orbs)), rng.standard_normal(20))
        acc.finalize()
        assert np.array_equal(acc.G, acc.G.T), (n_orbs, block)
    print("test_finalize_mirror_is_exactly_symmetric: OK")


def test_finalize_is_invariant_to_the_mirror_block_size():
    """Splitting the mirror differently may not perturb a single bit."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((40, 301))
    b = rng.standard_normal(40)
    ref = None
    for block in (1, 17, 300, 301, 4096):
        NormalEquationAccumulator._MIRROR_BLOCK = block
        acc = NormalEquationAccumulator(301)
        acc.add(A, b)
        out = acc.finalize()
        if ref is None:
            ref = out
        else:
            for got, want in zip(out, ref):
                assert np.array_equal(got, want), block
    print("test_finalize_is_invariant_to_the_mirror_block_size: OK")


@pytest.mark.parametrize("in_place", [False, True])
def test_finalize_does_not_allocate_a_second_full_matrix(in_place):
    """np.outer/np.triu_indices would each cost another n^2; the peak above
    the resident G must stay well under one extra full matrix (and near zero
    when P is built in G's own buffer)."""
    n = 1200
    NormalEquationAccumulator._MIRROR_BLOCK = 256
    rng = np.random.default_rng(3)
    acc = NormalEquationAccumulator(n)
    acc.add(rng.standard_normal((30, n)), rng.standard_normal(30))
    full = n * n * 8
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    P, q, col, b_max = acc.finalize(in_place=in_place)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    if in_place:
        assert peak - base < 0.25 * full, (peak - base, full)
    # P itself is one full matrix and is the returned value; anything beyond
    # ~1.25x that means a temporary of the kind we removed came back.
    assert peak - base < 1.25 * full, (peak - base, full)
    print(f"test_finalize_does_not_allocate_a_second_full_matrix: OK "
          f"(peak {(peak - base) / full:.2f}x one matrix)")


def test_matvec_f64_is_bit_exact_for_float64():
    rng = np.random.default_rng(4)
    for m, n in ((1, 1), (17, 5), (9000, 300), (300, 9000), (8192, 64), (8193, 64)):
        M = rng.standard_normal((m, n))
        v, w = rng.standard_normal(m), rng.standard_normal(n)
        assert np.array_equal(_matvec_f64(M, w), M @ w), (m, n)
        assert np.array_equal(_matvec_f64(M, v, transpose=True), M.T @ v), (m, n)
    print("test_matvec_f64_is_bit_exact_for_float64: OK")


def test_matvec_f64_matches_the_promoted_gemv_for_float32():
    """The chunked float32 path may differ from NumPy's whole-matrix
    promotion only by summation order."""
    rng = np.random.default_rng(5)
    for m, n in ((17, 5), (9000, 300), (300, 9000), (8192, 64), (8193, 64)):
        M = rng.standard_normal((m, n)).astype(np.float32)
        v, w = rng.standard_normal(m), rng.standard_normal(n)
        ref_n, ref_t = M @ w, M.T @ v
        # scale-relative, not elementwise: individual entries are dot
        # products that can cancel to near zero, where an elementwise rtol
        # measures nothing but the cancellation
        assert _rel(_matvec_f64(M, w), ref_n) < 1e-12, (m, n)
        assert _rel(_matvec_f64(M, v, transpose=True), ref_t) < 1e-12, (m, n)
    print("test_matvec_f64_matches_the_promoted_gemv_for_float32: OK")


def test_matvec_f64_is_stable_across_block_sizes():
    rng = np.random.default_rng(6)
    M = rng.standard_normal((1000, 40)).astype(np.float32)
    v, w = rng.standard_normal(1000), rng.standard_normal(40)
    ref_t, ref_n = M.T @ v, M @ w
    for block in (1, 7, 999, 1000, 10000):
        assert _rel(_matvec_f64(M, v, transpose=True, block=block), ref_t) < 1e-12, block
        # the non-transpose path writes disjoint slices, so only BLAS's own
        # choice of reduction for a narrower block can move it, and only at
        # round-off
        assert _rel(_matvec_f64(M, w, block=block), ref_n) < 1e-12, block
    print("test_matvec_f64_is_stable_across_block_sizes: OK")


def test_matvec_f64_does_not_promote_the_whole_float32_matrix():
    """A float32 M against a float64 v must not cost 2x sizeof(M)."""
    m, n, block = 20000, 400, 1024
    M = np.ones((m, n), dtype=np.float32)
    v = np.ones(m, dtype=np.float64)
    nbytes = M.nbytes
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    _matvec_f64(M, v, transpose=True, block=block)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # the whole-matrix promotion this replaced cost 2 * nbytes; the chunked
    # form must stay proportional to ONE promoted block instead, so the cost
    # no longer grows with the row count
    one_block = block * n * 8
    assert peak - base < 2.0 * one_block, (peak - base, one_block)
    assert peak - base < 0.5 * nbytes, (peak - base, nbytes)
    print(f"test_matvec_f64_does_not_promote_the_whole_float32_matrix: OK "
          f"(peak {(peak - base) / nbytes:.3f}x sizeof(M); the promoted gemv "
          f"it replaced cost 2.0x)")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("matvec + mirror, OK")
