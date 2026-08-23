# VERA Phase-1 Runbook (D1 smoke campaign)

Spec: `docs/superpowers/specs/2026-08-23-vera-evaluator-service-design.md`
Config: `dev_tests/vera_smoke_config.yaml` (production shape, `n_max_mods: 20`)

## 0. One-time layout

```bash
BASE=/vera/ptmp/gc/mia/pesmith/oCen
mkdir -p $BASE/{envs,runs/prod,archive}
git clone -b slurm https://github.com/pjs902/dynamite.git $BASE/dynamite
rsync -av --exclude='models/' \
    /nexus/posix0/MIA-astro-env/nneum/pesmith/PM_grid/NGC5139_production_input_xeast/ \
    $BASE/runs/prod/NGC5139_production_input_xeast/
cp $BASE/dynamite/dev_tests/vera_smoke_config.yaml $BASE/runs/prod/
```

## 1. Environment build

```bash
cd $BASE/dynamite
bash scripts/vera_env_setup.sh          # idempotent; see section 7 of spec
source $BASE/envs/dynamite/bin/activate
python -c "import dynamite, adelie; print('env OK')"
cat $BASE/ENV_FREEZE.txt                # record versions in this runbook's appendix
```

If MPCDF modules are preferable to conda for the numeric stack, build a venv
against the module python instead; the decision rule is wheels-over-modules
and whatever avoids the `/u` file quota.

## 2. Launch the driver

```bash
export VERA_CONFIG=$BASE/runs/prod/vera_smoke_config.yaml
export VERA_RUN_DIR=$BASE/runs/prod
cd $VERA_RUN_DIR

# dry cycle first: prints submission plan, touches nothing
python -m dynamite.vera.driver --config $VERA_CONFIG \
       --run-dir $VERA_RUN_DIR --dry-run --once

# for real (nohup so it survives logout)
LOG=driver_$(date +%Y%m%d_%H%M%S).log
nohup python -m dynamite.vera.driver --config $VERA_CONFIG \
       --run-dir $VERA_RUN_DIR > "$LOG" 2>&1 &
```

Expected first cycles: one `ocen-int` array (20 pending libraries packed
into ~2 node-tasks) plus, as sentinels age in, `ocen-solve` arrays capped at
`%16`. The driver writes `vera_inflight.json`, `vera_attempts.json`, and
`vera_ledger_jids.json` next to the config.

## 3. Kill/restart drill (acceptance requirement)

After at least one wave is submitted and jobs are running:

```bash
pkill -f "dynamite.vera.driver"        # driver only; array jobs keep running
# relaunch exactly as in step 2
```

Pass criteria:
- relaunch performs **zero** resubmission of work whose Slurm jobs are alive;
- after jobs finish, completions are observed and recorded normally.

## 4. Acceptance gate checklist (D1)

- [ ] all 20 table rows reach `all_done=True`
- [ ] end-to-end wall <= 24 h on <= 8 nodes
- [ ] fiducial-parset chi2 within 1e-6 relative of the local reference row
      (compare against
      `/nexus/.../PM_grid/NGC5139_production_output/all_models.ecsv`,
      matched on parset values)
- [ ] kill/restart drill passed with zero duplicate submissions
- [ ] `mmlsquota` guard and nightly tar verified (dry-run mode)

Record measured numbers below as they land.

## Appendix A - measured results

| quantity | value | date |
|---|---|---|
| env build time | TBD | |
| integration wave wall | TBD | |
| median solve wall (20 models) | TBD | |
| driver restart duplicates | expected 0 | |

## Appendix B - known local-box caveat

On the KVM dev box, importing torch before pymc trips a libstdc++/ICU ABI
clash (`CXXABI_1.3.15`), so BO-proposer tests skip locally
(`test_vera_proposer_bayesopt.py`). The clean VERA conda env does not have
this problem; if it ever appears there, pin `libstdcxx-ng` to match torch's
bundled runtime inside the env.
