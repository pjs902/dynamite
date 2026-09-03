# sBH profile fits — provenance for the Zhao component choice

Scripts behind `docs/superpowers/specs/2026-09-01-sbh-component-design.md`.
Run with the `main` conda env (`/opt/miniconda3/envs/main/bin/python`),
which has `gcfit` installed. Run from this directory, in order:

```
python target_phaseflow.py    # -> target_phaseflow.npz  (PhaseFlow fiducial BH density + M)
python target_gcfit.py        # -> target_gcfit.npz      (GCfit LIMEPY rho_BH + cum_M_BH)
python fit_joint.py           # variant table, scored on rho AND M(<r)
python fit_alternatives.py    # single vs double Zhao vs MGE
python check_closed_form.py   # M(<r) closed form vs quadrature  (asserts)
python check_potential.py     # hyp2f1 forms, incl. gamma > 2     (asserts, see note)
python plot_bestfit.py        # figures
```

`fit_families.py` and `plot_joint.py` are the earlier density-only scoring,
kept because the density-only and joint results differ in which variants
look acceptable.

## External inputs (not in this repo)

- `~/research/phaseflow/omegacen/output_pdmf/omegacen_pdmf*` plus
  `omegacen_pdmf.ini` — the fiducial PDMF run. Note this run used
  `hmax=1e10`, which truncates at r ~ 7 pc; the fitted scale radius exceeds
  the fitted range, so PhaseFlow constrains `gamma` and essentially nothing
  else.
- `~/research/omegaCen/BH_profiles/NGC5139_sampler.hdf` — the GCfit nested
  sampling run, read via `gcfit.analysis.NestedRun(...).get_CImodel()`.

## Notes

- `check_potential.py` intentionally ends in a failing assert: it
  demonstrates that the potential's outer term genuinely diverges for
  gamma >= 2 (`inf`/`nan` for alpha > 3, gamma = 2.24). That is real physics,
  not a bug, and is why the design tabulates Phi rather than restricting
  gamma. Read the printed table, not the exit code.
- The reference quadrature in these scripts integrates in log r. A naive
  `quad(..., 0, r)` or `quad(..., r, inf)` fails on the steep profiles and
  produced spurious disagreements on the first pass.
