"""Two figures:
 (1) the inner-mass-error figure, with the middle panel now showing the
     all-free Zhao best fit overlaid on BOTH external M(<r) curves;
 (2) a dedicated look at the all-free Zhao best fit to LIMEPY alone.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fit_joint import fit, zhao_rho, zhao_menc, VARIANTS, TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
IMBH = [8000.0, 50000.0]
FREE = "al,b,g all free"


def fitted(r, rho, M, vname=FREE):
    """Joint fit, returned as callables normalised to the target's M(<r_max)."""
    f = fit(r, rho, M, VARIANTS[vname])
    p = (1.0, f["a"], f["al"], f["b"], f["g"])
    rho0 = M[-1] / zhao_menc(r, *p)[-1]
    return f, (lambda x: zhao_rho(x, rho0, *p[1:])), (lambda x: zhao_menc(x, *p) * rho0)


pf_r, pf_rho, pf_M = TARGETS["PhaseFlow BH, 1e-4-5 pc"]
li_r, li_rho, li_M = TARGETS["LIMEPY, full 1e-3-50 pc"]
ob_r, ob_rho, ob_M = TARGETS["LIMEPY, 0.05-10 pc"]

f_pf, rho_pf, M_pf = fitted(pf_r, pf_rho, pf_M)
f_li, rho_li, M_li = fitted(li_r, li_rho, li_M)
f_ob, rho_ob, M_ob = fitted(ob_r, ob_rho, ob_M)

lab = lambda f: r"Zhao fit: $\gamma$=%.2f $\alpha$=%.2f $\beta$=%.2f a=%.2g pc" % (
    f["g"], f["al"], f["b"], f["a"])

# ======================= FIGURE 1 =====================================
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

ax = axes[0]
for vname, c in zip(["gNFW (legacy 5: al=1,b=3)", "al=2, b=4", FREE],
                    ["tab:red", "tab:blue", "tab:green"]):
    _, _, Mf = fitted(li_r, li_rho, li_M, vname)
    ax.semilogx(li_r, Mf(li_r) - li_M, lw=1.7, color=c, label=vname)
for m, ls in zip(IMBH, ["--", ":"]):
    ax.axhline(m, color="k", ls=ls, lw=1); ax.axhline(-m, color="k", ls=ls, lw=1)
    ax.annotate(r"$\pm M_\bullet$=%g" % m, (1.2e-3, m), fontsize=8, va="bottom")
ax.set(xlabel="r [pc]", ylabel=r"$\Delta M_{\rm sBH}(<r)$  [M$_\odot$]", ylim=(-6e4, 6e4))
ax.set_title("Absolute mass error (LIMEPY target)", fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=.3)

ax = axes[1]
ax.loglog(pf_r, pf_M, color="tab:purple", lw=3, alpha=.45,
          label=r"PhaseFlow ($M_\bullet$=8000): BW cusp")
ax.loglog(pf_r, M_pf(pf_r), color="tab:purple", lw=1.3, ls="--",
          label="  " + lab(f_pf))
ax.loglog(li_r, li_M, color="tab:orange", lw=3, alpha=.45,
          label="GCfit LIMEPY (no IMBH): core")
ax.loglog(li_r, M_li(li_r), color="tab:orange", lw=1.3, ls="--",
          label="  " + lab(f_li))
for m, ls in zip(IMBH, ["--", ":"]):
    ax.axhline(m, color="k", ls=ls, lw=1)
ax.axvspan(1e-4, 0.1, color="0.88", zorder=0)
ax.set(xlim=(1e-4, 50), ylim=(1e-1, 3e5), xlabel="r [pc]",
       ylabel=r"$M_{\rm sBH}(<r)$ [M$_\odot$]")
ax.set_title("All-free Zhao fit to each external model\n"
             "(note: totals differ 17x, 1.07e4 vs 1.79e5)", fontsize=9)
ax.legend(fontsize=6.5, loc="lower right"); ax.grid(alpha=.3)

ax = axes[2]
for vname, c in zip(["gNFW (legacy 5: al=1,b=3)", "al=2, b=4", FREE],
                    ["tab:red", "tab:blue", "tab:green"]):
    _, _, Mf = fitted(pf_r, pf_rho, pf_M, vname)
    ax.semilogx(pf_r, Mf(pf_r) - pf_M, lw=1.7, color=c, label=vname)
for m, ls in zip(IMBH, ["--", ":"]):
    ax.axhline(m, color="k", ls=ls, lw=1); ax.axhline(-m, color="k", ls=ls, lw=1)
ax.set(xlabel="r [pc]", ylabel=r"$\Delta M_{\rm sBH}(<r)$  [M$_\odot$]", ylim=(-6e4, 6e4))
ax.set_title("Same, PhaseFlow cusp target", fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=.3)

fig.tight_layout()
o1 = os.path.join(HERE, "inner_mass_error_v2.png")
fig.savefig(o1, dpi=130)

# ======================= FIGURE 2: LIMEPY only =========================
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True,
                       gridspec_kw=dict(height_ratios=[2.5, 1]))
curves = [("fit over full range", f_li, rho_li, M_li, "tab:green"),
          ("fit over 0.05-10 pc only", f_ob, rho_ob, M_ob, "tab:brown")]

ax[0, 0].loglog(li_r, li_rho, "k-", lw=4, alpha=.3, label="LIMEPY target")
ax[0, 1].loglog(li_r, li_M, "k-", lw=4, alpha=.3, label="LIMEPY target")
for name, f, rf, Mf, c in curves:
    ax[0, 0].loglog(li_r, rf(li_r), lw=1.6, color=c, label="%s\n  %s" % (name, lab(f)))
    ax[0, 1].loglog(li_r, Mf(li_r), lw=1.6, color=c, label=name)
    ax[1, 0].semilogx(li_r, np.log10(rf(li_r) / li_rho), lw=1.6, color=c)
    ax[1, 1].semilogx(li_r, Mf(li_r) - li_M, lw=1.6, color=c)
ax[0, 1].axhline(8000, color="k", ls="--", lw=1)
ax[0, 1].annotate(r"$M_\bullet$=8000", (1.2e-3, 8500), fontsize=8)
for m, ls in zip(IMBH, ["--", ":"]):
    ax[1, 1].axhline(m, color="k", ls=ls, lw=1); ax[1, 1].axhline(-m, color="k", ls=ls, lw=1)
ax[1, 1].annotate(r"$\pm M_\bullet$=8000", (1.2e-3, 9000), fontsize=8)
ax[1, 0].axhspan(-.0212, .0212, color="0.8", alpha=.6, zorder=0)
ax[0, 0].set_ylabel(r"$\rho_{\rm BH}$ [M$_\odot$ pc$^{-3}$]")
ax[0, 1].set_ylabel(r"$M_{\rm BH}(<r)$ [M$_\odot$]")
ax[1, 0].set_ylabel(r"$\log_{10}$ ratio $\rho$"); ax[1, 0].set_ylim(-.8, .8)
ax[1, 1].set_ylabel(r"$\Delta M$ [M$_\odot$]"); ax[1, 1].set_ylim(-6e4, 6e4)
for a in ax.ravel():
    a.grid(alpha=.3)
for a in ax[1]:
    a.set_xlabel("r [pc]"); a.axhline(0, color="k", lw=.8)
ax[0, 0].legend(fontsize=7, loc="lower left"); ax[0, 1].legend(fontsize=8, loc="upper left")
fig.suptitle("All-free Zhao best fit to the GCfit LIMEPY sBH profile", fontsize=11)
fig.tight_layout()
o2 = os.path.join(HERE, "limepy_bestfit.png")
fig.savefig(o2, dpi=130)

print("\nfit params:")
for n, f in [("PhaseFlow", f_pf), ("LIMEPY full", f_li), ("LIMEPY 0.05-10", f_ob)]:
    print("  %-16s g=%.3f al=%.3f b=%.3f a=%.4g pc   rho %.3f dex  M %.3f dex"
          % (n, f["g"], f["al"], f["b"], f["a"], f["rms_rho"], f["rms_M"]))
print("\nLIMEPY, worst |dM| by radial zone (full-range fit / observable-range fit):")
for lo, hi in [(1e-3, .1), (.1, 1), (1, 10), (10, 50)]:
    s = (li_r >= lo) & (li_r <= hi)
    print("  %5.3g-%-5.3g pc:  %9.0f  /  %9.0f  Msun"
          % (lo, hi, np.max(np.abs(M_li(li_r)[s] - li_M[s])),
             np.max(np.abs(M_ob(li_r)[s] - li_M[s]))))
print(o1); print(o2)
