"""Extract the GCfit (LIMEPY) rho_BH median + 68% CI for NGC 5139."""
import os
import numpy as np
import gcfit, gcfit.analysis

HERE = os.path.dirname(os.path.abspath(__file__))
BH = "/Users/pesmith/research/omegaCen/BH_profiles"
os.chdir(BH)

obs = gcfit.Observations("NGC5139")
nested = gcfit.analysis.NestedRun("./NGC5139_sampler.hdf", observations=obs)
ci = nested.get_CImodel(load=True)

r = ci.r.to_value("pc")
rho = np.asarray(ci.rho_BH.value)          # (1, 5, Nr): -2s,-1s,med,+1s,+2s
cum = np.asarray(ci.cum_M_BH.value)
np.savez(os.path.join(HERE, "target_gcfit.npz"),
         r=r, rho=rho[0], cum=cum[0], BH_mass=np.asarray(ci.BH_mass.value))
print("r: %.3e .. %.3e pc  (N=%d)" % (r[0], r[-1], len(r)))
med = rho[0, 2]
good = med > 0
print("rho_BH median: %.3e .. %.3e" % (med[good][0], med[good][-1]))
print("total M_BH (median cum):", cum[0, 2][-1])
print("M_BH samples: mean %.4g  16/50/84 %s"
      % (np.mean(ci.BH_mass.value), np.percentile(ci.BH_mass.value, [16, 50, 84])))
rr, mm = r[good], med[good]
for lo, hi in [(1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1), (1, 3)]:
    s = (rr >= lo) & (rr <= hi)
    if s.sum() > 3:
        print("  log-slope %.3g-%.3g pc: %.3f"
              % (lo, hi, np.polyfit(np.log(rr[s]), np.log(mm[s]), 1)[0]))
