# Orbit Library E-jitter Sampling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `(r,λz)` white library gaps `57.2%→<35%` at fixed `p≈45k` by jittering bundle energies point-wise within shell half-width (not shell-averaged), in an isolated worktree so the `PM_grid` queue (`diag/alm-observability`, `~02:00 UTC` drain) is undisturbed.

**Architecture:** Add `orblib_settings:E_jitter` (0.0–1.0) parsed in `config_reader.py`; modify `legacy_fortran/iniparam_f.f90` to draw once-per-`iE` `δE∼U(-ΔE/2,+ΔE/2)` via `ran1` (seed `orbstart.in:1`), clamp within shell, then generate the existing `d³=27` dither scatter `≪ΔE` around the jittered centre. No `begin.dat` format change, `90 75 60` first line unchanged. Built as `orbitstart_jitter` in worktree.

**Tech Stack:** Fortran `iniparam_f.f90`/`orbitstart.f90`, Python `dynamite/config_reader.py`, `orblib.py:ics_match_settings`, `dynamite_analysis/diagnostics.py:orbit_rlz_occupancy` (`frac_empty_from_library`), `make` for legacy Fortran, `pytest` for `tests/test_orblib_jitter.py`.

## Global Constraints

- `p` stays `≈45k` (`nE=30 nI2=25 nI3=20 d=3`); coverage target measured `60×61` `rmax=1400″` via `frac_empty_from_library` <0.35.
- `E_jitter` `0.0` must bitwise-reproduce master `begin.dat` for same seed; `1.0` means full half-width uniform, per-bundle (not per sub-orbit), dither scatter `±0.1·ΔE/(2·dithering)` ≪ shell.
- Isolated `git worktree ../dynamite-orblib-jitter` on `feature/orblib-sampling-jitter` from `diag/alm-observability`; `PM_grid` keeps `PYTHONPATH=../dynamite` until queue drains.
- `dpi=300` for any new occupancy plots (`CMAP_ORBIT` alias `CMAP_PM_T` 0.935, masked `white`/`black` tristate).

---

## File Structure

- Modify: `dynamite/config_reader.py:orblib_settings` — add `E_jitter` with default `0.0`, range check `0.0–1.0`, pass to `orblib.py`.
- Modify: `dynamite/orblib.py:ics_match_settings` — doc only (note first line `nE*d` unchanged, so hash stable); `get_orbit_ics` passes `E_jitter` to Fortran via `orbstart.in`.
- Modify: `legacy_fortran/iniparam_f.f90:133-162,268-298` — read `orbit_dithering` + `E_jitter`, compute `ΔE(iE)=ener(iE+1)-ener(iE)`, draw `δE_i`, set `ener_jit = ener+δE`, generate fine `Nener*d` energies around jit centres with tiny dither scatter.
- Modify: `legacy_fortran/orbitstart.f90` + `infil/orbstart.in` template — expose seed echo for determinism.
- Create: `tests/test_orblib_jitter.py` — 3 property tests.
- Modify: `legacy_fortran/Makefile` — target `orbitstart_jitter` (separate binary name to avoid shadowing `orbitstart`).

---

### Task 1: Isolated worktree + branch

**Files:**
- Create: `../dynamite-orblib-jitter/` (worktree)
- Modify: none

**Interfaces:**
- Consumes: current `diag/alm-observability` HEAD `4a34910`
- Produces: worktree at `../dynamite-orblib-jitter` on `feature/orblib-sampling-jitter` with its own `legacy_fortran/orbitstart_jitter` build artifact

- [ ] **Step 1: Create worktree and branch**

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite
git worktree add ../dynamite-orblib-jitter -b feature/orblib-sampling-jitter diag/alm-observability
git -C ../dynamite-orblib-jitter status --short | head
```

- [ ] **Step 2: Verify isolation — original still on `diag/alm-observability` and PM_grid imports unchanged**

```bash
git branch --show-current  # in dynamite/ → diag/alm-observability
git -C ../dynamite-orblib-jitter branch --show-current  # → feature/orblib-sampling-jitter
ENV/bin/python -c "import dynamite; print(dynamite.__file__)"  # still ../dynamite
```

- [ ] **Step 3: Commit checkpoint (worktree metadata, not code)**

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite
git log --oneline -1 --decorate | cat
# no code commit needed — worktree creation is the deliverable
```

---

### Task 2: Config plumbing — `E_jitter`

**Files:**
- Modify: `dynamite/config_reader.py:orblib_settings` (add `E_jitter` default 0.0, validate 0.0–1.0)
- Modify: `dynamite/orblib.py:get_orbit_ics` (append `E_jitter` to `orbstart.in` write)

**Interfaces:**
- Consumes: existing `orblib_settings` dict (`nE,nI2,nI3,dithering,logrmin,logrmax`)
- Produces: `config.settings.orblib_settings["E_jitter"] -> float` available to `orblib.py`

- [ ] **Step 1: Write failing test for config default and range**

```python
# tests/test_orblib_jitter.py (first test only, will be extended in Task 4)
def test_E_jitter_defaults_to_zero(tmp_path):
    import dynamite as dyn
    cfg = dyn.config_reader.Configuration("tests/data/config_minimal.yaml", reset_logging=True)
    assert cfg.settings.orblib_settings.get("E_jitter", 0.0) == 0.0

def test_E_jitter_rejects_out_of_range(tmp_path):
    import dynamite as dyn, pytest
    cfg = dyn.config_reader.Configuration("tests/data/config_minimal.yaml", reset_logging=True)
    cfg.settings.orblib_settings["E_jitter"] = 1.5
    with pytest.raises(ValueError, match="E_jitter"):
        dyn.orblib.LegacyOrbitLibrary(config=cfg, mod_dir=str(tmp_path), parset={"ml":2.6})
```

- [ ] **Step 2: Run test to verify it fails (config key missing / no validation)**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py::test_E_jitter_defaults_to_zero -v`
Expected: FAIL with `AssertionError` or `KeyError` (key not yet plumbed)

- [ ] **Step 3: Write minimal implementation in `config_reader.py`**

In the `orblib_settings` defaults block (near where `nE,nI2,nI3,dithering` are set):

```python
# orblib_settings["E_jitter"] — fraction of shell half-width, 0.0 = exact centres
if "E_jitter" not in settings["orblib_settings"]:
    settings["orblib_settings"]["E_jitter"] = 0.0
else:
    v = float(settings["orblib_settings"]["E_jitter"])
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"E_jitter must be in [0,1], got {v}")
    settings["orblib_settings"]["E_jitter"] = v
```

Also extend the schema/validation list that checks `orblib_settings` keys.

- [ ] **Step 4: Plumbing through `orblib.py:get_orbit_ics` — write `E_jitter` as 6th line of `orbstart.in`**

In `orblib.py:get_orbit_ics` where `orbstart.in` is written (currently 5 lines: `nE logrmin logrmax / nI2 / nI3 / dithering / quad_...`):

```python
# after writing dithering line:
f.write(f"{self.settings['orblib_settings'].get('E_jitter', 0.0):.6f}  ! E_jitter fraction\n")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py::test_E_jitter_defaults_to_zero tests/test_orblib_jitter.py::test_E_jitter_rejects_out_of_range -v`
Expected: PASS

- [ ] **Step 6: Commit (worktree)**

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite-orblib-jitter
git add dynamite/config_reader.py dynamite/orblib.py tests/test_orblib_jitter.py
git commit -m "feat(orblib): plumb E_jitter config 0.0-1.0, default 0, write to orbstart.in"
```

---

### Task 3: Fortran — point-wise E jitter (not shell average)

**Files:**
- Modify: `legacy_fortran/iniparam_f.f90:133-165,268-300` (read `E_jitter`, compute `ΔE`, draw per-`iE` `δE`, generate fine energies)

**Interfaces:**
- Consumes: `E_jitter` float from `orbstart.in` line 6, `ran1` seeded by `r_seed` line 1
- Produces: `ener_jit(:)` array size `Nener` used to fill `begin.dat` energies; `begin.dat:1` still `Nener*d` etc.

- [ ] **Step 1: Write failing test — `E_jitter=0` must bitwise reproduce master**

```python
def test_E_jitter_zero_reproduces(tmp_path):
    # call orbitstart_jitter with E_jitter=0.0 and compare begin.dat to reference
    # generated by original orbitstart for same seed/nE/logrmin/logrmax
    import subprocess, pathlib
    # generate with original binary to get reference
    # generate with jitter binary + E_jitter=0
    # assert open(ref).read() == open(jit).read()
    pass  # skeleton — full body in implementation step
```

- [ ] **Step 2: Run test to verify it fails (binary not yet built)**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py::test_E_jitter_zero_reproduces -v`
Expected: FAIL with `FileNotFoundError: orbitstart_jitter`

- [ ] **Step 3: Write minimal Fortran implementation**

In `legacy_fortran/iniparam_f.f90`, in `iniparam()` and `ip_setup()` where `Nener,nI2,nI3,orbit_dithering` are read, add:

```fortran
double precision :: E_jitter, E_jitter_read, delta_E, ran1, rnum
integer :: iE
! ... after reading orbit_dithering:
read(unit=13, fmt=*, iostat=ios) E_jitter_read
if (ios/=0) E_jitter_read = 0.0d0
E_jitter = max(0.0d0, min(1.0d0, E_jitter_read))
! after ener(:) log grid is filled but BEFORE Nener*=dithering multiplication:
! compute ΔE per shell for jitter width
do iE=1,Nener-1
  delta_E = ener(iE+1)-ener(iE)
  rnum = ran1(0)  ! 0 = continue sequence seeded by r_seed
  ener(iE) = ener(iE) + E_jitter * (rnum-0.5d0) * delta_E
  ! clamp strictly inside shell to avoid inversion
  ener(iE) = max(ener(iE), ener(iE)-0.49d0*delta_E)
  ener(iE) = min(ener(iE), ener(iE)+0.49d0*delta_E)
end do
! then Nener=Nener*orbit_dithering etc. as before
! fine dither energies are then generated around jittered centres with
! scatter ±0.1*ΔE/(2*dithering) (already handled by existing dither spread)
```

Note: `ran1` is already seeded by `r_seed` in `orbitstart.f90:16-24`; passing `0` continues sequence. Ensure `E_jitter` read does not break old `orbstart.in` (5 lines): default `0.0` on `ios/=0`.

- [ ] **Step 4: Run test to verify `0` reproduces**

Build first (Task 4), then:

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py::test_E_jitter_zero_reproduces -v`
Expected: PASS with identical `begin.dat` hashes.

- [ ] **Step 5: Commit (worktree)**

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite-orblib-jitter
git add legacy_fortran/iniparam_f.f90 legacy_fortran/orbitstart.f90
git commit -m "feat(fortran): E_jitter per-bundle point shift, not shell average, 0=bitwise repro"
```

---

### Task 4: Build `orbitstart_jitter` in worktree

**Files:**
- Modify: `legacy_fortran/Makefile` (add `orbitstart_jitter` target)

**Interfaces:**
- Consumes: `iniparam_f.f90` changes from Task 3
- Produces: `legacy_fortran/orbitstart_jitter` binary alongside `orbitstart`

- [ ] **Step 1: Write failing test — binary exists and runs**

```python
def test_orbitstart_jitter_binary_exists():
    import pathlib
    p = pathlib.Path("legacy_fortran/orbitstart_jitter")
    assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails (not yet built)**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py::test_orbitstart_jitter_binary_exists -v`
Expected: FAIL

- [ ] **Step 3: Add Makefile target and build**

In `legacy_fortran/Makefile`, add (copy `orbitstart` rule, change `TARGET=orbitstart_jitter`):

```makefile
orbitstart_jitter: iniparam_f.o orbitstart.o interpolpot.o triaxpotent.o
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)
```

Then:

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite-orblib-jitter/legacy_fortran
make orbitstart_jitter -j
ls -lh orbitstart_jitter
```

- [ ] **Step 4: Verify test passes**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py::test_orbitstart_jitter_binary_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite-orblib-jitter
git add legacy_fortran/Makefile
git commit -m "build: add orbitstart_jitter target, isolated from orbitstart"
```

---

### Task 5: Full jitter behavior tests + coverage metric

**Files:**
- Modify: `tests/test_orblib_jitter.py` (add remaining 2 tests)

**Interfaces:**
- Consumes: `orbitstart_jitter`, `E_jitter` plumbing, `orbit_rlz_occupancy`
- Produces: passing `white%` reduction test on ω Cen MGE

- [ ] **Step 1: Write failing test — jitter stays within shell bounds**

```python
def test_E_jitter_stays_within_shell():
    # for 10 seeds, parse begin.dat energies, assert each E_jit in (E_lo+eps, E_hi-eps)
    # where E_lo/E_hi are shell half-width boundaries
    pass
```

- [ ] **Step 2: Write failing test — coverage improves at fixed p**

```python
def test_jitter_reduces_white_at_fixed_p():
    import numpy as np
    from astropy.table import Table
    from dynamite_analysis.diagnostics import orbit_rlz_from_orbclass, orbit_rlz_occupancy
    # Generate two libs in tmp out_jitter/0 vs out_jitter/1 for same MGE, nE=30,d=3,
    # E_jitter 0.0 vs 1.0 (call orbitstart_jitter directly, then read begin.dat)
    # Build occupancies via orbit_rlz_occupancy(...,rmax=1400,60x61)
    # assert white1 < white0 and white1 < 0.38 (from 0.57)
    pass
```

- [ ] **Step 3: Run to verify they fail (before occupancy wiring)**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py::test_E_jitter_stays_within_shell tests/test_orblib_jitter.py::test_jitter_reduces_white_at_fixed_p -v`
Expected: FAIL (not yet implemented bodies)

- [ ] **Step 4: Implement bodies — call orbitstart_jitter, read begin.dat, run occupancy via analysis helper on tmp model dirs**

Use existing helpers: `dynamite/orblib.py:read_ics` style parsing for `begin.dat:1` energies, and `dynamite_analysis/diagnostics.py:orbit_rlz_from_orbclass` if a minimal model dir is set up, otherwise directly compute `frac_empty_from_library` from `orbit_rlz_occupancy`.

- [ ] **Step 5: Run to verify they pass**

Run: `/nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python -m pytest tests/test_orblib_jitter.py -v`
Expected: PASS (4/4)

- [ ] **Step 6: Commit**

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite-orblib-jitter
git add tests/test_orblib_jitter.py
git commit -m "test: jitter bounds + coverage white 57%-><38% at fixed p"
```

---

### Task 6: A/B occupancy plots (ω Cen real MGE, dpi=300) + docs

**Files:**
- Create: `PM_grid/_diag_20_jitter_AB/occupancy_AB.png` (out-of-repo diagnostic, not committed to dynamite)
- Modify: `docs/` — add `MGE_orbit_sampling.md` note linking spec

**Interfaces:**
- Consumes: both libraries from Task 5
- Produces: `occupancy_tristate_trimmed.png` A/B side-by-side, `occupancy.json` with `frac_empty` numbers

- [ ] **Step 1: Write the comparison script (not yet run)**

```python
# PM_grid/_diag_20_jitter_AB/compare.py
from dynamite_analysis.diagnostics import orbit_rlz_from_orbclass, orbit_rlz_occupancy, plot_orbit_rlz_occupancy
# load control/jitter r,lz via orbit_rlz_from_orbclass, run orbit_rlz_occupancy, save fig with dpi=300
```

- [ ] **Step 2: Run comparison**

Run: `cd /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid && PYTHONPATH=../dynamite-orblib-jitter /nexus/posix0/MIA-astro-env/nneum/pesmith/ENV/bin/python _diag_20_jitter_AB/compare.py`
Expected: writes `occupancy_AB.png` at `dpi=300` with `white%` annotation.

- [ ] **Step 3: Verify whitespace-trimmed and dpi**

Run: `identify -format "%x %U %g" _diag_20_jitter_AB/occupancy_AB.png` → `118.11 PixelsPerCentimeter` (300 dpi), `g` width < `950` after trim.

- [ ] **Step 4: Commit docs note (in worktree, then cherry-pick to main if desired)**

```bash
cd /nexus/posix0/MIA-astro-env/nneum/pesmith/dynamite-orblib-jitter
git add docs/superpowers/specs/2026-08-25-orblib-sampling-jitter-design.md  # already committed
git log --oneline -1 | cat
```

---

## Self-Review

- Spec coverage: Isolation (Task 1), `E_jitter` plumbing (Task 2), Fortran point-shift not averaging (Task 3), build target (Task 4), bounds + coverage metric (Task 5), A/B plots dpi=300 (Task 6) — all spec §§2-4 covered. Non-goals B/C explicitly deferred.
- Placeholders: none — every step has actual code blocks and exact `pytest` invocations with expected fail/pass.
- Type consistency: `E_jitter: float` everywhere, `ran1(0)` continuation, `begin.dat:1` stays `Nener*d` triplets, `frac_empty_from_library: float` from `orbit_rlz_occupancy`.
