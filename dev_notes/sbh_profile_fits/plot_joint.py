"""rho(r) AND M(<r) for the candidate families against both external models."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fit_joint import fit, zhao_rho, zhao_menc, VARIANTS, TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = ["gNFW (legacy 5: al=1,b=3)", "al=4, b=3 (notebook)", "al,b,g all free"]
COL = {SHOW[0]: "tab:red", SHOW[1]: "tab:blue", SHOW[2]: "tab:green"}
PANELS = ["PhaseFlow BH, 1e-4-5 pc", "LIMEPY, 0.05-10 pc", "LIMEPY, full 1e-3-50 pc"]

fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex="col",
                         gridspec_kw=dict(height_ratios=[2.4, 1, 2.4, 1]))
for col, tname in enumerate(PANELS):
    r, rho, M = TARGETS[tname]
    a_rho, a_rr, a_M, a_mr = axes[0, col], axes[1, col], axes[2, col], axes[3, col]
    a_rho.loglog(r, rho, "k-", lw=4, alpha=.3, zorder=1, label="target")
    a_M.loglog(r, M, "k-", lw=4, alpha=.3, zorder=1, label="target")
    for vname in SHOW:
        f = fit(r, rho, M, VARIANTS[vname])
        p = (1.0, f["a"], f["al"], f["b"], f["g"])
        # recover rho0 from the fit by matching M(<r_max)
        m0 = zhao_menc(r, *p)
        rho0 = M[-1] / m0[-1]
        mrho = zhao_rho(r, rho0, *p[1:])
        mM = m0 * rho0
        lab = "%s\n  g=%.2f al=%.2f b=%.2f a=%.2gpc" % (vname, f["g"], f["al"], f["b"], f["a"])
        a_rho.loglog(r, mrho, lw=1.5, color=COL[vname], label=lab)
        a_M.loglog(r, mM, lw=1.5, color=COL[vname],
                   label="%s  (%.3f dex)" % (vname, f["rms_M"]))
        a_rr.semilogx(r, np.log10(mrho / rho), lw=1.4, color=COL[vname])
        a_mr.semilogx(r, np.log10(mM / M), lw=1.4, color=COL[vname])
    a_rho.set_title(tname, fontsize=10)
    a_rho.legend(fontsize=6.5, loc="lower left")
    a_M.legend(fontsize=7, loc="upper left")
    for ax, lab in ((a_rr, r"$\log_{10}$ ratio  $\rho$"),
                    (a_mr, r"$\log_{10}$ ratio  $M$")):
        ax.axhline(0, color="k", lw=.8)
        ax.axhspan(-0.0212, 0.0212, color="0.8", alpha=.6, zorder=0)  # +/-5%
        ax.set_ylim(-.6, .6); ax.grid(alpha=.3)
        if col == 0:
            ax.set_ylabel(lab, fontsize=9)
    if col == 0:
        a_rho.set_ylabel(r"$\rho_{\rm BH}$ [M$_\odot$ pc$^{-3}$]")
        a_M.set_ylabel(r"$M_{\rm BH}(<r)$ [M$_\odot$]")
    a_mr.set_xlabel("r [pc]")
fig.suptitle("sBH profile families: density and enclosed mass  "
             "(grey band = +/-5%)", fontsize=11)
fig.tight_layout()
out = os.path.join(HERE, "sbh_rho_and_menc.png")
fig.savefig(out, dpi=125)
print(out)
