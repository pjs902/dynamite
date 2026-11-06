"""In-memory pieces of ``pm_recentre_mode: vp`` (NNLS._solve_relative_vp).

vp's objective over the weights w, with the relative PM sets' per-aperture
centres c profiled out:

    Phi(w) = min_c J(w, c),
    J(w, c) = other(w) + sum_a ||(Sx(c_a) H_a(w) Sy(c_a)' - D_a) / E_a||^2
              + lam ||w col0||^2,

H_a(w) = sum_o w_o h_oa (the unshifted orbit histograms), other(w) = every
row but the relative PM rows and the total-mass row (the QP's equality),
as a quadratic form. With the orbit histograms held as one sparse matrix
per set (captured during vp's first library read) and other(w) from pass
0's Gram, Phi and its gradient (envelope theorem: the gradient of J at the
fitted c) cost two sparse products and a batched centre fit, seconds
instead of a library read. :class:`VPPolish` runs L-BFGS-B on Phi from
vp's result; :meth:`VPPolish.data_ridge` gives the ridge's (lam, col0)
from the data alone.

Validated against the full-scale i44_vp run (2026-10-28): the same J
reproduces its logged Phi to 1e-4, and L-BFGS-B on it converges to one
point from the pass-0 weights, uniform weights and vp's endpoint
(relative_proper_motions/inram/).
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy import optimize
from scipy.interpolate import CubicSpline

from dynamite import kinematics as dyn_kin


class ShiftMatrices:
    """``kinematics.pm_shift_matrix(edg, m)`` for many m at once.

    pm_shift_matrix regrids the cumulative distribution with a not-a-knot
    spline on the shifted edges ``edg - m``; that spline is the one on
    ``edg`` evaluated at ``x + m``, so one spline serves every shift.
    Returns (len(m), n, n)."""

    def __init__(self, edg):
        self.edg = np.asarray(edg, dtype=float)
        n = len(self.edg) - 1
        self.spline = CubicSpline(self.edg, np.eye(n + 1), bc_type="not-a-knot")
        self.cum = np.tril(np.ones((n + 1, n)), -1)

    def __call__(self, m, nder=0):
        """S, or with ``nder`` = 2 the tuple (S, dS/dm, d2S/dm2). Clipping
        the edges to the window makes S piecewise smooth, with kinks where
        an edge crosses the window's end (whole-cell shifts on a uniform
        grid); exactly on one the derivatives are the mean of the two sides,
        as central differences give."""
        e = self.edg
        x = e[None, :] + np.asarray(m, dtype=float)[:, None]
        xc = np.clip(x, e[0], e[-1])
        r = np.diff(self.spline(xc) @ self.cum, axis=1)
        col = r.sum(axis=1, keepdims=True)
        pos = col > 0
        S = np.divide(r, col, out=r.copy(), where=pos)
        if nder == 0:
            return S
        inside = (((x > e[0]) & (x < e[-1])) + 0.5 * ((x == e[0]) | (x == e[-1])))[:, :, None]
        r1 = np.diff((self.spline(xc, 1) * inside) @ self.cum, axis=1)
        r2 = np.diff((self.spline(xc, 2) * inside) @ self.cum, axis=1)
        c1, c2 = r1.sum(axis=1, keepdims=True), r2.sum(axis=1, keepdims=True)
        # S = r / col (r itself where col <= 0), differentiated
        S1 = np.divide(r1 - S * c1, col, out=r1.copy(), where=pos)
        S2 = np.divide(r2 - 2 * S1 * c1 - S * c2, col, out=r2.copy(), where=pos)
        return S, S1, S2


class CentreFit:
    """Batched version of ``NNLS._vp_fit_centres`` for one set: per aperture
    a, the centre c minimising f_a(c) = sum(((Sx(c) H_a Sy(c)') - D_a) / E_a)^2.

    Levenberg-Marquardt on the exact derivatives, all apertures at
    once (Nelder-Mead per aperture in a Python loop takes ~80 s for 1404
    apertures; this takes about a second). Starts and scan as there: from
    c0, and with ``scan`` also from the best point of a +-3-cell grid
    (half-cell steps) around c0, keeping the lower end."""

    def __init__(self, xedg, D, E):
        self.sx, self.sy = ShiftMatrices(xedg[0]), ShiftMatrices(xedg[1])
        self.D, self.inv = D, 1.0 / E
        self.dv = min(np.diff(xedg[0]).min(), np.diff(xedg[1]).min())

    def chi2(self, H, c, idx=slice(None)):
        R = (self.sx(c[:, 0]) @ H[idx] @ self.sy(c[:, 1]).transpose(0, 2, 1)
             - self.D[idx]) * self.inv[idx]
        return np.einsum("axy,axy->a", R, R)

    def fit(self, H, c0, scan):
        """Returns (c (n_ap, 2), f (n_ap,))."""
        c0 = np.asarray(c0, dtype=float)
        c, f = self._lm(H, c0.copy())
        if scan:
            grid = np.arange(-6, 7) * 0.5 * self.dv
            best, arg = np.full(len(c0), np.inf), np.zeros((len(c0), 2))
            Sy = [self.sy(c0[:, 1] + g).transpose(0, 2, 1) for g in grid]
            for gx in grid:
                SxH = self.sx(c0[:, 0] + gx) @ H
                for gy, sy in zip(grid, Sy):
                    R = (SxH @ sy - self.D) * self.inv
                    fg = np.einsum("axy,axy->a", R, R)
                    better = fg < best
                    best[better] = fg[better]
                    arg[better] = (gx, gy)
            c2, f2 = self._lm(H, c0 + arg)
            better = f2 < f
            c[better], f[better] = c2[better], f2[better]
        return c, f

    def derivatives(self, H, c, idx=slice(None)):
        """f_a(c) and its exact gradient (n_ap, 2) and Hessian (hxx, hyy,
        hxy), from the shift matrices' spline derivatives."""
        Sx, Sx1, Sx2 = self.sx(c[:, 0], 2)
        Sy, Sy1, Sy2 = (S.transpose(0, 2, 1) for S in self.sy(c[:, 1], 2))
        Ha, inv = H[idx], self.inv[idx]
        A0, A1 = Sx @ Ha, Sx1 @ Ha
        R = (A0 @ Sy - self.D[idx]) * inv
        Jx, Jy = (A1 @ Sy) * inv, (A0 @ Sy1) * inv

        def dot(P, Q):
            return np.einsum("axy,axy->a", P, Q)
        g = 2 * np.stack([dot(R, Jx), dot(R, Jy)], -1)
        hxx = 2 * (dot(Jx, Jx) + dot(R, (Sx2 @ Ha @ Sy) * inv))
        hyy = 2 * (dot(Jy, Jy) + dot(R, (A0 @ Sy2) * inv))
        hxy = 2 * (dot(Jx, Jy) + dot(R, (A1 @ Sy1) * inv))
        return dot(R, R), g, hxx, hyy, hxy

    def _lm(self, H, c, max_iter=200, ftol=1e-6):
        """Levenberg-Marquardt on the exact derivatives (full Hessian,
        shifted to positive definite). Stops an aperture once the undamped
        Newton step would lower f by less than ``ftol`` (chi2 units).
        Where f has a kink at its minimum (see ``ShiftMatrices``) that never
        happens; there it stops as before, on a tiny accepted step or on mu
        blowing up, since only steps that lower f are taken."""
        tol = 1e-7 * self.dv
        f = self.chi2(H, c)
        mu = np.full(len(c), 1e-3)
        active = np.ones(len(c), dtype=bool)
        for _ in range(max_iter):
            a = np.flatnonzero(active)
            if a.size == 0:
                break
            ca, fa = c[a], f[a]
            _, g, hxx, hyy, hxy = self.derivatives(H, ca, a)
            # shift the Hessian to positive definite, then damp
            lmin = 0.5 * (hxx + hyy) - np.hypot(0.5 * (hxx - hyy), hxy)
            scale = np.maximum(np.maximum(np.abs(hxx), np.abs(hyy)), 1e-300)
            s0 = np.maximum(0.0, -lmin) + 1e-12 * scale
            det0 = (hxx + s0) * (hyy + s0) - hxy ** 2
            gain0 = 0.5 * ((hyy + s0) * g[:, 0] ** 2 - 2 * hxy * g[:, 0] * g[:, 1]
                           + (hxx + s0) * g[:, 1] ** 2) / det0
            conv = gain0 < ftol
            s = np.maximum(0.0, -lmin) + mu[a] * scale
            axx, ayy = hxx + s, hyy + s
            det = axx * ayy - hxy ** 2
            d = -np.stack([ayy * g[:, 0] - hxy * g[:, 1],
                           axx * g[:, 1] - hxy * g[:, 0]], -1) / det[:, None]
            # never jump more than half a cell: other minima sit ~1 cell off
            n = np.hypot(d[:, 0], d[:, 1])
            d *= np.minimum(1.0, 0.5 * self.dv / np.maximum(n, 1e-300))[:, None]
            fn = self.chi2(H, ca + d, a)
            ok = fn < fa
            c[a[ok]] += d[ok]
            f[a[ok]] = fn[ok]
            mu[a] = np.where(ok, mu[a] / 3, mu[a] * 4)
            done = conv | (ok & (np.hypot(d[:, 0], d[:, 1]) < tol)) | (mu[a] > 1e12)
            active[a[done]] = False
        return c, f


class VPPolish:
    """Phi(w) and its gradient in memory, and L-BFGS-B on Phi.

    ``sets``: {name: dict(L, D, E, xedg)}, L the unshifted, unwhitened
    orbit histograms (n_orbs, n_ap * nvx * nvy) in aperture-major,
    row-major order, D and E (n_ap, nvx, nvy) the data and errors on their
    own grid (as vp has them). ``G``, ``v``, ``b_sq``: other(w) =
    w'Gw - 2 w'v + b_sq + sum over ``rot`` of (||A_new w - b||^2 -
    ||A_old w - b||^2), i.e. pass 0's rows with its rotation rows (``A_old``)
    swapped for vp's (``A_new``). ``M``: the total mass, held exactly.
    """

    def __init__(self, sets, G, v, b_sq, rot, M, logger):
        self.sets = {}
        for nm, s in sets.items():
            L = s["L"].tocsc()
            n_ap, nvx, nvy = s["D"].shape
            self.sets[nm] = dict(L=L, LT=L.T.tocsr(copy=False), D=s["D"],
                                 E=s["E"], shape=(n_ap, nvx, nvy),
                                 fit=CentreFit(s["xedg"], s["D"], s["E"]))
        self.G, self.v, self.b_sq, self.rot, self.M = G, v, b_sq, rot, M
        self.logger = logger
        self.lam, self.col0 = 0.0, np.ones(len(v))

    def data_ridge(self, m_ref, ridge_lambda, dtype):
        """(lam, col0) of the ridge from the data alone: col0 = sqrt(diag)
        of the raw Gram of pass 1 of the other modes (relative data and
        errors shifted to the start centres ``m_ref``, {name: (n_ap, 2)}, a
        data-template shift basis projected out, as
        ``NNLS._relative_pm_block`` builds it), lam = ridge_lambda * (number
        of nonzero columns) / n_orbs, which is what ``_apply_diagonal_ridge``
        adds to that build's normalised P. No weights enter, so Phi is the
        same function whatever the start."""
        t0 = time.time()
        diag = np.diag(self.G).copy()
        for A_old, A_new, _ in self.rot:
            diag += (A_new ** 2).sum(axis=0) - (A_old ** 2).sum(axis=0)
        for nm, s in self.sets.items():
            n_ap, nvx, nvy = s["shape"]
            k = nvx * nvy
            xedg = (s["fit"].sx.edg, s["fit"].sy.edg)
            vx, vy = 0.5 * (xedg[0][1:] + xedg[0][:-1]), 0.5 * (xedg[1][1:] + xedg[1][:-1])
            Ds, Es = dyn_kin.ProperMotions.shifted_data(s["D"], s["E"], xedg,
                                                        -np.asarray(m_ref[nm]))
            L = s["L"]
            for a in range(n_ap):
                e = Es[a].ravel()
                G = dyn_kin.pm_shift_basis(Ds[a][None], vx, vy)[0]
                G = G.reshape(-1, 2) / e[:, None]
                G_pinv = np.linalg.pinv(G)
                h = L[:, a * k:(a + 1) * k].toarray().astype(dtype) / e.astype(dtype)
                h -= (h @ G_pinv.T.astype(dtype)) @ G.T.astype(dtype)
                diag += np.einsum("ok,ok->o", h, h, dtype=np.float64)
        col0 = np.sqrt(np.abs(diag))
        nz = col0 != 0
        col0[~nz] = 1.0
        lam = ridge_lambda * nz.sum() / len(col0)
        self.logger.info(f"relative PM: vp ridge from the data: lam {lam:.6g}, "
                         f"{nz.sum()} nonzero columns, {time.time() - t0:.0f} s")
        return lam, col0

    def other(self, w):
        Gw = self.G @ w
        val = float(w @ Gw - 2 * w @ self.v + self.b_sq)
        g = Gw - self.v
        for A_old, A_new, b in self.rot:
            rn, ro = A_new @ w - b, A_old @ w - b
            val += float(rn @ rn - ro @ ro)
            g += A_new.T @ rn - A_old.T @ ro
        return val, g

    def phi(self, w, c, scan, grad=True):
        """Phi(w) with the centres refitted from c ({name: (n_ap, 2)});
        returns dict(phi, other, pm, ridge, c, g (dPhi/dw))."""
        oth, g = self.other(w)
        out = dict(other=oth, pm=0.0, ridge=self.lam * float(((w * self.col0) ** 2).sum()),
                   c={})
        for nm, s in self.sets.items():
            H = (s["LT"] @ w).reshape(s["shape"])
            cf, _ = s["fit"].fit(H, c[nm], scan)
            out["c"][nm] = cf
            Sx, Sy = s["fit"].sx(cf[:, 0]), s["fit"].sy(cf[:, 1])
            R = (Sx @ H @ Sy.transpose(0, 2, 1) - s["D"]) / s["E"]
            out["pm"] += float((R * R).sum())
            if grad:
                g += s["L"] @ (Sx.transpose(0, 2, 1) @ (R / s["E"]) @ Sy).ravel()
        out["phi"] = out["other"] + out["pm"] + out["ridge"]
        if grad:
            out["g"] = 2 * (g + self.lam * self.col0 ** 2 * w)
        return out

    def run(self, w, c, maxfun, tol, window):
        """L-BFGS-B on Phi from (w, c), w >= 0 and sum(w) = M exactly
        (w = M x / sum x, x = u / col0, u >= 0: column-scaled weights, so
        the Hessian diagonal is ~2 in u). Stops when the best Phi has
        fallen by less than ``tol`` over the last ``window`` evaluations,
        on L-BFGS-B's own convergence, or after ``maxfun`` evaluations.
        Every evaluation refits the centres from the last ones (the first
        with the scan); the best point gets a final scan refit.

        Returns (w, c, history {phi: best Phi after each evaluation}, stop).
        """
        col0, M = self.col0, self.M
        st = dict(c=c, best=np.inf, w=w, cb=c, n=0, hist=[])

        class Done(Exception):
            pass

        def w_of(u):
            x = u / col0
            return M * x / x.sum(), x.sum()

        def fg(u):
            wv, sx = w_of(u)
            r = self.phi(wv, st["c"], scan=st["n"] == 0)
            st["n"] += 1
            st["c"] = r["c"]
            if r["phi"] < st["best"]:
                st.update(best=r["phi"], w=wv, cb=r["c"])
            st["hist"].append(st["best"])
            h = st["hist"]
            if len(h) > window and h[-window - 1] - h[-1] < tol:
                raise Done
            g = r["g"]
            return r["phi"], (M / sx) * (g - (g @ wv) / M) / col0

        u0 = w * col0
        pos = u0 > 0
        u0 = u0 / (np.median(u0[pos]) if pos.any() else 1.0)
        t0 = time.time()
        try:
            res = optimize.minimize(fg, u0, jac=True, method="L-BFGS-B",
                                    bounds=[(0, None)] * len(u0),
                                    options=dict(maxfun=maxfun, maxiter=maxfun,
                                                 maxcor=20, ftol=1e-15, gtol=0))
            stop = "maxfun" if res.status == 1 else f"lbfgsb_{res.status}"
        except Done:
            stop = "tol"
        r = self.phi(st["w"], st["cb"], scan=True, grad=False)
        if r["phi"] < st["best"]:
            st.update(best=r["phi"], cb=r["c"])
            st["hist"].append(r["phi"])
        self.logger.info(f"relative PM: vp polish ({stop}): {st['n']} evaluations, "
                         f"Phi {st['hist'][0]:.8g} -> {st['best']:.8g}, "
                         f"{time.time() - t0:.0f} s")
        return st["w"], st["cb"], dict(phi=st["hist"]), stop
