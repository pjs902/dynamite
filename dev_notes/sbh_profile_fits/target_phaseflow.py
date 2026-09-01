"""Extract the PhaseFlow fiducial PDMF run's summed BH-component density."""
import os, re, glob, configparser
import numpy as np

PF = "/Users/pesmith/research/phaseflow/omegacen"
prefix = os.path.join(PF, "output_pdmf", "omegacen_pdmf")
ini = os.path.join(PF, "omegacen_pdmf.ini")
HERE = os.path.dirname(os.path.abspath(__file__))

cfg = configparser.ConfigParser(); cfg.read(ini)
secs = [s for s in cfg.sections() if s != "PhaseFlow"]
suf2type = {chr(ord('a') + i): s.split()[0] for i, s in enumerate(secs)}
bh_suf = [s for s, t in suf2type.items() if t == "BH"]
cap = max(float(cfg[s].get("captureRadius", 0)) for i, s in enumerate(secs)
          if suf2type[chr(ord('a') + i)] == "BH")

rs, rhos = [], []
for suf in bh_suf:
    files = [f for f in glob.glob(f"{prefix}*{suf}") if not f.endswith(".log")]
    times = []
    for f in files:
        m = re.match(re.escape(prefix) + r"(\d+)" + suf + r"$", f)
        if m:
            times.append(int(m.group(1)))
    d = np.loadtxt(prefix + str(max(times)) + suf, skiprows=2)
    rs.append(d[:, 0]); rhos.append(d[:, 3])

r_lo = max(r.min() for r in rs); r_hi = min(r.max() for r in rs)
rg = np.geomspace(r_lo, r_hi, 400)
rho_tot = np.sum([np.exp(np.interp(np.log(rg), np.log(r), np.log(np.maximum(rho, 1e-300))))
                  for r, rho in zip(rs, rhos)], axis=0)

m = (rg > 3 * cap) & (rg < 5.0) & (rho_tot > 0)
r_pf, rho_pf = rg[m], rho_tot[m]
np.savez(os.path.join(HERE, "target_phaseflow.npz"), r=r_pf, rho=rho_pf)
print("BH components:", len(bh_suf), " captureRadius:", cap)
print("pts %d  r=%.3e..%.3f pc  rho=%.3e..%.3e" %
      (len(r_pf), r_pf[0], r_pf[-1], rho_pf[0], rho_pf[-1]))
print("inner log-slope (first 40 pts):",
      np.polyfit(np.log(r_pf[:40]), np.log(rho_pf[:40]), 1)[0])
print("outer log-slope (last 40 pts):",
      np.polyfit(np.log(r_pf[-40:]), np.log(rho_pf[-40:]), 1)[0])
