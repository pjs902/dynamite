#!/usr/bin/env python3
"""Seed-replicated landscape comparison: is the BayesOpt advantage stable?

Runs each (landscape, arm) across N seeds and reports median [min, max] of
models-to-threshold and best chi2. Single-seed results prove possibility;
this tests reliability.

Run: python dev_tests/run_seed_sweep.py [n_seeds]
"""

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chi2_landscapes as L  # noqa: E402
import run_surface_comparison as R  # noqa: E402


def best_and_count(y, threshold):
    ok = np.isfinite(y)
    if not np.any(ok):
        return float("inf"), None
    hit = ok & (y <= threshold)
    return float(np.min(y[ok])), (int(np.argmax(hit) + 1) if np.any(hit) else None)


def main(n_seeds=6):
    rows = defaultdict(list)
    for land in [L.QML_RIDGE, L.PRODUCTION, L.HALO_BANANA]:
        for arm in ["gridwalk", "bayesopt"]:
            for seed in range(n_seeds):
                r = R.run_arm(land, arm, seed=seed)
                y = np.asarray(r["y"], dtype=float)
                best, cnt = best_and_count(y, land.threshold)
                rows[(land.name, arm)].append(
                    {"n": len(y), "best": best, "to_thr": cnt, "stalled": bool((r["stalled_at"] or 0) > 0)}
                )
                print(f"{land.name:<14} {arm:<9} s{seed}: n={len(y):>3} best={best:7.1f} thr@{cnt}", flush=True)

    print(f"\n=== {n_seeds} seeds: median [min..max] ===")
    print(f"{'landscape':<14} {'arm':<10} {'models':>16} {'best_chi2':>20} {'thr hit':>8}")
    verdict = {}
    for land in [L.QML_RIDGE, L.PRODUCTION, L.HALO_BANANA]:
        summary = {}
        for arm in ["gridwalk", "bayesopt"]:
            rr = rows[(land.name, arm)]
            ns = [r["n"] for r in rr]
            bs = [r["best"] for r in rr]
            hits = sum(1 for r in rr if r["to_thr"] is not None)
            m2t = sorted(r["to_thr"] for r in rr if r["to_thr"] is not None)
            med = m2t[len(m2t) // 2] if m2t else None
            summary[arm] = (med, hits)
            print(
                f"{land.name:<14} {arm:<10} "
                f"{np.median(ns):>6.0f} [{min(ns)}..{max(ns)}]"
                f" {np.median(bs):>10.0f} [{min(bs):.0f}..{max(bs):.0f}]"
                f" {hits:>3}/{n_seeds}" + (f"  median-to-thr={med}" if med is not None else "")
            )
        gw, bo = summary["gridwalk"], summary["bayesopt"]
        verdict[land.name] = bo[1] > gw[1] or (bo[1] == gw[1] == n_seeds and (bo[0] or 10**9) < (gw[0] or 10**9))
    print("\nverdicts (BayesOpt more reliable or faster to threshold):")
    for k, v in verdict.items():
        print(f"  {k:<14} {'BAYESOPT' if v else 'GRIDWALK'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
