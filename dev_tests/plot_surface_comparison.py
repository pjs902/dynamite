#!/usr/bin/env python3
"""Trajectory + convergence figures for the surface comparison.

Reads dev_tests/surface_runs/*.npz (from run_surface_comparison.py) and
writes PNGs to dev_tests/surface_figs/.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/pesmith/.opencode/skills/figure-style")
import kernel  # noqa: E402
import chi2_landscapes as L  # noqa: E402

kernel.apply_figure_style(sizes=(8, 7, 6))
CMAP_TIME = "viridis"  # shared semantics: dark=early, bright=late

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "surface_runs")
FIGS = os.path.join(HERE, "surface_figs")
os.makedirs(FIGS, exist_ok=True)

# which projections to show per landscape (row = (x-axis idx, y-axis idx))
PROJ = {
    "shape-ridge": [(0, 1), (1, 2)],  # (ml,q) tilt; (q,p) banana
    "production-4d": [(0, 1), (2, 3)],  # (BH,ml); (q,p) band
    "halo-banana": [(2, 0), (1, 2)],  # (f,ml) valley; (c,f) flat+basin
}
DISPLAY = {
    "ml": "$M_\\mathrm{L}$",
    "q": "$q$",
    "p": "$p$",
    "log10 MBH": "$\\log_{10}M_\\mathrm{BH}$",
    "c-dh": "$c_\\mathrm{dh}$",
    "log f-dh": "$\\log\\,f_\\mathrm{dh}$",
}


def load(name, arm):
    d = np.load(os.path.join(RUNS, f"{name}_{arm}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def bbox_check(fig, fname):
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(plt.Text) if t.get_text().strip() and t.get_visible()]
    overlaps = [
        (a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1 :] if ba.overlaps(bb)
    ]
    if overlaps:
        print(f"  !! text overlaps in {fname}: {overlaps[:5]}")
    return not overlaps


def trajectory_fig(land):
    arms = [("gridwalk", "grid walk"), ("bayesopt", "BayesOpt")]
    proj = PROJ[land.name]
    fig, axes = plt.subplots(len(proj), 2, figsize=(6.3, 2.6 * len(proj)), squeeze=False)
    data = {arm: load(land.name, arm) for arm, _ in arms}

    # per-row axis limits follow the DATA envelope (both arms), not the box
    lims = []
    for ia, ib in proj:
        xs = np.concatenate([data[a]["X"][:, ia] for a, _ in arms])
        ys = np.concatenate([data[a]["X"][:, ib] for a, _ in arms])
        padx = 0.06 * (xs.max() - xs.min())
        pady = 0.10 * (ys.max() - ys.min())
        lims.append(
            (
                max(land.bounds[ia][0], xs.min() - padx),
                min(land.bounds[ia][1], xs.max() + padx),
                max(land.bounds[ib][0], ys.min() - pady),
                min(land.bounds[ib][1], ys.max() + pady),
            )
        )

    for c, (arm, label) in enumerate(arms):
        d = data[arm]
        X, y = d["X"], d["y"]
        for r, (ia, ib) in enumerate(proj):
            ax = axes[r][c]
            G, H, Z = land.grid(n=140, axes=(ia, ib), other=X[y.argmin()])
            zmin = Z.min()
            levels = np.linspace(zmin + 8, zmin + 750, 12)
            ax.contourf(G, H, Z, levels=levels, cmap="Grays", alpha=0.55)
            ax.contour(G, H, Z, levels=[land.threshold], colors=["#b2182b"], linewidths=1.2)
            ax.scatter(
                X[:, ia],
                X[:, ib],
                c=np.arange(len(X)),
                cmap=CMAP_TIME,
                s=14,
                lw=0.4,
                edgecolors="k",
                zorder=3,
            )
            if arm == "gridwalk" and len(X) > 1:
                ax.plot(X[:, ia], X[:, ib], color="0.35", lw=0.7, alpha=0.65, zorder=2)
            xb = X[y.argmin()]
            ax.plot(xb[ia], xb[ib], marker="*", ms=11, mfc="#fde725", mec="k", mew=0.6, ls="none", zorder=4)
            ax.set_xlabel(DISPLAY[land.names[ia]])
            if c == 0:
                ax.set_ylabel(DISPLAY[land.names[ib]])
            ax.set_xlim(lims[r][0], lims[r][1])
            ax.set_ylim(lims[r][2], lims[r][3])
            kernel.set_frame(ax)
            ax.tick_params(direction="out", top=False, right=False, labelsize=6, which="both")
            ax.set_xlabel(DISPLAY[land.names[ia]], labelpad=9)
            if c == 0:
                ax.set_ylabel(DISPLAY[land.names[ib]], labelpad=9)
            if r == 0 and c == 0:
                ax.text(
                    0.03,
                    0.05,
                    "red = convergence threshold · star = best found",
                    transform=ax.transAxes,
                    fontsize=6,
                    va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
                )

    # headers carry method identity -> no legend needed
    for c, (arm, label) in enumerate(arms):
        d = data[arm]
        y = d["y"]
        tag = f"{label} — {len(y)} models · best $\\chi^2$={np.nanmin(y):.0f}"
        if d["stalled_at"] > 0:
            tag += " · stalled"
        axes[0][c].set_title(tag, fontsize=8, pad=6)
    fig.suptitle(
        f"{land.name}: {land.desc}   (point shade = evaluation order, dark $\\to$ bright)",
        x=0.02,
        ha="left",
        fontsize=8,
    )
    fig.subplots_adjust(wspace=0.24, hspace=0.42, right=0.985, top=0.80, bottom=0.12)
    # hide tick labels that graze a neighbouring axis's label at a corner
    for ax in fig.axes:
        for xy, lim_get in [("x", ax.get_xlim), ("y", ax.get_ylim)]:
            axis = getattr(ax, f"{xy}axis")
            lo, hi = lim_get()
            ticks = axis.get_ticklocs()
            if not len(ticks):
                continue
            step = np.median(np.diff(ticks)) if len(ticks) > 1 else 1.0
            labels = axis.get_ticklabels()
            if abs(ticks[0] - lo) < 0.35 * step and labels:
                labels[0].set_visible(False)
            if abs(ticks[-1] - hi) < 0.35 * step and labels:
                labels[-1].set_visible(False)

    fname = os.path.join(FIGS, f"traj_{land.name}.png")
    fig.savefig(fname, dpi=300)
    ok = bbox_check(fig, fname)
    plt.close(fig)
    print(f"wrote {fname}" + ("" if ok else "  [OVERLAPS]"))


def convergence_fig():
    lands = [L.QML_RIDGE, L.PRODUCTION, L.HALO_BANANA]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    for ax, land in zip(axes, lands):
        gw = load(land.name, "gridwalk")
        bo = load(land.name, "bayesopt")
        for d, color, lbl in [
            (gw, "#2166ac", "grid walk"),
            (bo, "#e08214", "BayesOpt"),
        ]:
            y = d["y"]
            cum = np.minimum.accumulate(np.nan_to_num(y, nan=np.inf))
            xs = np.arange(1, len(cum) + 1)
            ax.plot(xs, cum, color=color, lw=1.4, label=lbl)
            ax.axhline(land.threshold, color="#b2182b", lw=0.8, ls="--")
            ax.plot(xs[-1], cum[-1], marker="o", ms=3, color=color)
        ax.set_title(land.name, fontsize=8)
        ax.set_xlabel("models evaluated")
        ax.margins(x=0.04, y=0.12)
        ax.tick_params(direction="out", top=False, right=False, labelsize=6, which="both")
        kernel.set_frame(ax)
    axes[0].set_ylabel("$\\chi^2$ (best so far)")
    handles = [
        Line2D([], [], color="#2166ac", lw=1.4, label="grid walk"),
        Line2D([], [], color="#e08214", lw=1.4, label="BayesOpt"),
        Line2D([], [], color="#b2182b", lw=0.8, ls="--", label="convergence threshold"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.34, bottom=0.32)
    fname = os.path.join(FIGS, "convergence.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


if __name__ == "__main__":
    for land in [L.QML_RIDGE, L.PRODUCTION, L.HALO_BANANA]:
        trajectory_fig(land)
    convergence_fig()
