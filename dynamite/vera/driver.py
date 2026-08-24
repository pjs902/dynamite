"""VERA driver daemon: scan, reconcile, submit, observe (spec section 5.4).

Single writer of ``all_models.ecsv`` (atomic replace). All state beyond the
artifacts lives in two JSON ledgers next to the run directory:

* ``vera_attempts.json``  - {model_dir: attempts}
* ``vera_inflight.json``  - {kind: [item, ...]} currently-submitted work

The driver is restart-safe: every phase is a pure function of filesystem +
Slurm state plus these ledgers.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time

import numpy as np

from .classifier import ATTEMPT_LIMIT, WEIGHTS, ModelState, _noml, classify
from .pack_integration import pack_libraries
from . import SCHEMA_VERSION
from .proposal import Result, validate_parset
from .slurm import (
    MAX_ARRAY_SIZE,
    SlurmError,
    _current_user,
    atomic_write_json,
    build_integration_job_spec,
    build_solve_job_spec,
    levelfs,
    run_argv,
    running_job_ids,
    submit_array,
    write_manifest,
)

POLL_INTERVAL_S = 300
K_START = 16
ATTEMPTS_FILE = "vera_attempts.json"
LEDGER_FILE = "vera_inflight.json"
DIRS_FILE = "vera_dirs.json"



class VeraDriver:
    def __init__(
        self,
        config,
        proposer,
        runner=None,
        run_dir=".",
        poll_interval=POLL_INTERVAL_S,
        k_start=K_START,
        clock=time.time,
    ):
        self.clock = clock
        self.config = config
        self.proposer = proposer
        self.runner = runner or run_argv
        self.run_dir = os.path.abspath(run_dir)
        self.poll_interval = poll_interval
        self.k_start = k_start
        self.attempts = self._load_json(ATTEMPTS_FILE, {})
        # {kind: {item: array job id}} -- in-flight membership IS the key set,
        # so the two cannot drift the way separate files could
        self.inflight = self._load_json(LEDGER_FILE, {"int": {}, "solve": {}})
        self.output_root = config.settings.io_settings["output_directory"]
        # proposal attribution, owned by the driver (not the proposer):
        dirs_map = self._load_json(DIRS_FILE, {})
        self.dir_to_pid = dirs_map.get("dir_to_pid", {})
        # parked models already announced to the proposer; persisted so a
        # restart does not replay every past failure
        self.reported_failed = set(dirs_map.get("reported_failed", []))
        self.log = logging.getLogger(f"{__name__}.VeraDriver")

    # ------------------------------------------------------------ persistence
    def _load_json(self, name, default):
        p = os.path.join(self.run_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        return default

    def _dump_json(self, name, obj):
        atomic_write_json(os.path.join(self.run_dir, name), obj)

    def _save_dirs(self):
        self._dump_json(DIRS_FILE, {"dir_to_pid": self.dir_to_pid,
                                    "reported_failed": sorted(self.reported_failed)})

    def _save_table_atomically(self):
        tbl = self.config.all_models.table
        rel = self.config.settings.io_settings.get("all_models_file", "all_models.ecsv")
        target = os.path.join(self.output_root, rel)
        fd, tmp = tempfile.mkstemp(
            suffix=".ecsv", dir=os.path.dirname(os.path.abspath(target))
        )
        os.close(fd)
        try:
            tbl.write(tmp, format="ascii.ecsv", overwrite=True)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------------------------------------------------ phases
    def _dir_to_row(self):
        """directory -> table row index, derived fresh; the table is the
        source of truth and a cached copy goes stale across restarts."""
        t = self.config.all_models.table
        return {str(r["directory"]): i for i, r in enumerate(t) if r["directory"]}

    def scan(self):
        """{model_dir: ModelState} for every row in the table."""
        states = {}
        now = self.clock()
        for row in self.config.all_models.table:
            d = row["directory"]
            if not d or str(d).startswith("rejected"):
                continue
            full = os.path.join(self.output_root, "models", d)
            states[d] = classify(full, attempts=self.attempts.get(d, 0), now_ts=now)
        return states

    def reconcile_and_submit(self, dry_run=False, states=None):
        """Returns the number of Slurm ARRAY JOBS submitted this cycle (not
        the number of models: one array carries many)."""
        states = self.scan() if states is None else states
        if dry_run:
            live = set()
        else:
            try:
                live = running_job_ids(self.runner)
            except SlurmError as e:
                # a flaky squeue must not look like "every job died", which
                # would resubmit the whole campaign on top of itself
                self.log.warning("squeue failed (%s); skipping this cycle", e)
                return 0
        charged = set()
        for kind in ("int", "solve"):
            live_items = {}
            for item, jid in dict(self.inflight.get(kind, {})).items():
                if jid is None:
                    # no recorded job: treat as dead, not live. Stalling is the
                    # worse failure; a needless resubmit is bounded by
                    # ATTEMPT_LIMIT and short-circuits on the artifacts.
                    self.log.warning("no ledger jid for %s %s; assuming it died", kind, item)
                elif jid in live:
                    live_items[item] = jid
                    continue
                # the job is gone. Charge unless THIS kind of work left its
                # artifact: an int job succeeds at TO_SOLVE, a solve job only
                # at SOLVED.
                succeeded = (
                    (ModelState.TO_SOLVE, ModelState.SOLVED)
                    if kind == "int"
                    else (ModelState.SOLVED,)
                )
                if states.get(item) not in succeeded:
                    self._charge_attempt(item)
                    charged.add(item)

            self.inflight[kind] = live_items
        self._dump_json(LEDGER_FILE, self.inflight)
        if charged:
            self._dump_json(ATTEMPTS_FILE, self.attempts)
            # patch rather than re-scan: only the charged rows can have changed
            # state, and only to PARKED (classify checks attempts right after
            # the weights file). A full re-scan re-stats the whole campaign.
            for d in charged:
                if (self.attempts.get(d, 0) >= ATTEMPT_LIMIT
                        and states.get(d) is not ModelState.SOLVED):
                    states[d] = ModelState.PARKED

        # One integration per LIBRARY, not per model: ml variants share a
        # library (and its noml directory), so submitting each of them would
        # race several processes into the same datfil. The siblings pick up
        # the sentinel and go straight to TO_SOLVE on a later cycle.
        to_int, seen_libs = [], set()
        for d in sorted(
            d for d, s in states.items() if s is ModelState.PENDING_INTEGRATION
        ):
            lib = _noml(os.path.join(self.output_root, "models", d))
            if lib in seen_libs:
                continue
            seen_libs.add(lib)
            to_int.append(d)
        to_sol = sorted(d for d, s in states.items() if s is ModelState.TO_SOLVE)

        rows = self._dir_to_row()
        for d in list(to_int) + list(to_sol):
            self._write_parset_file(d, rows)
        self.log.debug("states=%s to_int=%s to_sol=%s inflight=%s",
                       {k: v.value for k, v in states.items()},
                       to_int, to_sol, self.inflight)
        submitted = 0
        if to_int:
            submitted += self._submit_wave("int", pack_libraries(to_int), dry_run)
        if to_sol:
            # a built library always earns its weights: chi2 feeds every
            # future proposer decision. The quorum gate lives on PROPOSING
            # new work (step()), never on draining existing artifacts.
            submitted += self._submit_wave("solve", [[d] for d in to_sol], dry_run)
        return submitted

    def _submit_wave(self, kind, packages, dry_run):
        """`packages` groups model dirs into one array task each.

        Membership is tracked per MODEL DIR, never per joined package
        string: a wave repacks whenever its composition changes, which
        gives a still-running model a brand-new package string and used to
        read as new work -- resubmitting a live integration.
        """
        done = set(self.inflight.get(kind, {}))
        fresh = [[d for d in pkg if d not in done] for pkg in packages]
        fresh = [pkg for pkg in fresh if pkg]
        if not fresh:
            return 0
        if dry_run:
            flat = [d for pkg in fresh for d in pkg]
            print(
                f"[dry-run] would submit {len(flat)} {kind} model(s) in "
                f"{len(fresh)} package(s): {flat[:3]}{'...' if len(flat) > 3 else ''}"
            )
            return 0
        if kind == "int":
            script = os.path.join(
                os.path.dirname(__file__), "scripts", "integrate_package.sh"
            )
        else:
            script = os.path.join(os.path.dirname(__file__), "scripts", "solve_task.sh")
        # a wave larger than the scheduler's MaxArraySize goes out as several
        # arrays: truncating would leave the tail recorded as in-flight and
        # never scheduled, so those models would never run and never retry
        n_arrays = 0
        for chunk in [fresh[i:i + MAX_ARRAY_SIZE] for i in range(0, len(fresh), MAX_ARRAY_SIZE)]:
            items = [";".join(pkg) for pkg in chunk]
            if kind == "int":
                spec = build_integration_job_spec(n_items=len(chunk))
            else:
                # fairshare-adaptive throttle, asked for only when we
                # actually have solve work to submit
                spec = build_solve_job_spec(k=self._adaptive_k(), n_items=len(chunk))
            if hasattr(self.runner, "submit_array"):
                # local backend: synchronous, but same manifest/index contract
                jid = self.runner.submit_array(script, items)
            else:
                manifest = write_manifest(self.run_dir, kind, items)
                jid = submit_array(self.runner, spec, script, items, manifest)
            for pkg in chunk:
                for d in pkg:
                    self.inflight.setdefault(kind, {})[d] = jid
            print(f"submitted {kind} array job {jid} with {len(chunk)} task(s)")
            n_arrays += 1
        self._dump_json(LEDGER_FILE, self.inflight)
        return n_arrays

    def _charge_attempt(self, model_dir):
        n = int(self.attempts.get(model_dir, 0)) + 1
        self.attempts[model_dir] = n
        if n >= ATTEMPT_LIMIT:
            self.log.warning("%s parked after %d failed attempt(s)", model_dir, n)

    def _adaptive_k(self):
        try:
            lf = levelfs(self.runner, _current_user())
        except SlurmError as e:
            self.log.warning("sshare failed (%s); using default throttle", e)
            return self.k_start
        if lf is None:
            return self.k_start
        if lf > 10.0:
            return min(24, self.k_start + 4)
        if lf < 1.0:
            return max(4, self.k_start - 4)
        return self.k_start


    def _write_parset_file(self, model_dir, rows):
        """Drop <models>/<dir>/vera_parset.json next to the artifacts.

        Workers are pure functions of (config, parset): they never read the
        shared all_models table, because every Configuration init runs
        update_model_table() - whose janitor would delete *other* pending
        rows it finds there (concurrent readers are outside DYNAMITE's
        contract). Full par-values, fixed parameters included.
        """
        t = self.config.all_models.table
        i = rows.get(model_dir)
        if i is None:
            return

        payload = {"schema_version": SCHEMA_VERSION,
                   "par_names": list(self.config.parspace.par_names),
                   "values": {n: float(t[n][i])
                              for n in self.config.parspace.par_names}}
        target = os.path.join(self.output_root, "models", model_dir,
                              "vera_parset.json")
        # Rewrite only on change: this runs every cycle for work still
        # queued. Compare content, not existence, so a re-clipped row is not
        # left driven by a stale file.
        if os.path.isfile(target):
            try:
                with open(target) as f:
                    if json.load(f) == payload:
                        return
            except (OSError, ValueError):
                pass  # unreadable: fall through and rewrite it
        atomic_write_json(target, payload)

    def observe_completions(self, states=None):
        states = self.scan() if states is None else states
        results = []
        t = self.config.all_models.table
        rows = self._dir_to_row()
        for d, state in states.items():
            pid = self.dir_to_pid.get(d)
            row = rows.get(d)
            if pid is None or row is None:
                continue
            if state is ModelState.SOLVED and not bool(t["all_done"][row]):
                try:
                    meta = _read_weights_meta(os.path.join(self.output_root, "models", d))
                except Exception as e:
                    # a partially-flushed ecsv on NFS is readable next cycle;
                    # it is not a reason to lose the campaign
                    self.log.warning("weights for %s not readable yet (%s)", d, e)
                    continue
                results.append(
                    Result(proposal_id=pid, model_dir=d, status="done", **meta)
                )
                t["chi2"][row] = meta["chi2"]
                t["kinchi2"][row] = meta["kinchi2"]
                t["kinmapchi2"][row] = meta["kinmapchi2"]
                t["all_done"][row] = True
                t["weights_done"][row] = True
                t["time_modified"][row] = str(np.datetime64("now", "ms"))
        failed = [d for d, s in states.items() if s is ModelState.PARKED]
        for d in failed:
            pid = self.dir_to_pid.get(d)
            if pid is None or d in self.reported_failed:
                continue  # a parked model stays parked; report it once
            self.reported_failed.add(d)
            results.append(Result(proposal_id=pid, model_dir=d, status="failed"))
        if results:
            self.proposer.observe(results)
            self._save_table_atomically()
            self._save_dirs()
        return len(results)

    def exhausted(self):
        """Campaign finished according to the proposer's stopping rules."""
        return self.proposer.exhausted()

    def step(self, dry_run=False):
        # one filesystem snapshot for both phases: classify() is a pure
        # function of the filesystem plus self.attempts, and nothing between
        # them changes either, so a second scan re-stats the whole campaign
        # for the same answer
        states = self.scan()
        self.observe_completions(states)
        self.reconcile_and_submit(dry_run=dry_run, states=states)
        if self.proposer.ready_to_propose() and not self.exhausted():
            t = self.config.all_models.table
            before = len(t)
            self.proposer.propose()
            self._assign_directories(range(before, len(t)))
            # Fresh scan here, unlike the pair above: a synchronous backend
            # (LocalRunner) has actually RUN the work submitted a few lines
            # earlier, so the snapshot from the top of the cycle is stale by
            # now. Proposing cycles only happen when a batch has drained, so
            # this scan is rare; the savings are on the common path above.
            self.reconcile_and_submit(dry_run=dry_run)
        return not self.exhausted()

    # ------------------------------------------------------------ model rows
    def _assign_directories(self, row_indices):
        """Validate intake, then assign orblib_xxx_yyy/mlzz.zz/ directories.

        Rows come from the proposer's generate(); this method never appends
        rows - it names them exactly like
        ModelInnerIterator.assign_model_directories does, then records the
        dir->pid attribution the observer needs. Rejected proposals keep
        their row as an audit trail under rejected/<pid>/, are dropped from
        tracking, and surface as a failed Result.
        """
        t = self.config.all_models.table
        bounds = self._intake_bounds()
        qobs = self._qobs()
        shape_names = self._shape_param_names()
        u_fixed = self.proposer_u_fixed(shape_names)
        sformat = self.config.system.parameters[0].sformat
        orblib_cols, orblib_index = self._orblib_index(self._orblib_parameters())
        n_assigned = sum(
            1 for r in t
            if r["directory"] and not str(r["directory"]).startswith("rejected")
        )

        row_to_pid = {row: pid for pid, row in
                      self.proposer.pid_to_row.items()}

        for idx in row_indices:
            pid = row_to_pid.get(idx)
            if pid is None:
                continue
            # par_generator_settings lo/hi are RAW (log10 for logarithmic
            # params) while the table column holds par_value (physical).
            # Validate in raw space or the gate is a no-op for log params.
            pars = {p.name: p for p in self.config.parspace}
            parset = {
                name: float(pars[name].get_raw_value_from_par_value(float(t[name][idx])))
                for name in self.proposer.par_names
                if name in pars and name in t.colnames
            }
            clipped, violations = validate_parset(
                parset, bounds, qobs=qobs, u_fixed=u_fixed,
                shape_names=shape_names)
            if violations:
                t["directory"][idx] = f"rejected/{pid}/"
                self.proposer.reject(pid, violations)
                continue
            for name, raw in clipped.items():
                repaired = float(pars[name].get_par_value_from_raw_value(raw))
                if not np.isclose(repaired, float(t[name][idx])):
                    self.log.warning(
                        "intake clipped %s %s -> %g", pid, name, repaired
                    )
                    t[name][idx] = repaired
            iteration = int(t["which_iter"][idx])
            ml_val = float(t["ml"][idx]) if "ml" in t.colnames else 0.0
            # reuse the library of any earlier row sharing this potential:
            # an orbit library is ml-independent (as ModelInnerIterator does)
            reuse = self._existing_orblib_dir(idx, orblib_cols, orblib_index)
            if reuse is not None:
                directory = f"{reuse}/ml{ml_val:{sformat}}/"
            else:
                prefix = f"orblib_{iteration:03d}_{n_assigned:03d}"
                directory = f"{prefix}/ml{ml_val:{sformat}}/"
                orblib_index.append(
                    (np.array([float(t[c][idx]) for c in orblib_cols]), prefix)
                )
            t["directory"][idx] = directory
            self.dir_to_pid[directory] = pid
            n_assigned += 1
        self._save_table_atomically()
        self._save_dirs()

    def run_forever(self, dry_run=False, once=False, max_consecutive_errors=5):
        """The daemon outlives transient failures; it gives up only if it
        cannot complete several cycles in a row."""
        errors = 0
        while True:
            try:
                alive = self.step(dry_run=dry_run)
                errors = 0
            except Exception as e:  # NFS hiccups and half-written files too
                errors += 1
                self.log.warning("cycle failed (%d/%d): %s", errors, max_consecutive_errors, e)
                if errors >= max_consecutive_errors:
                    self.log.error("giving up after %d consecutive failures", errors)
                    raise
                alive = True
            if once or not alive:
                break
            time.sleep(self.poll_interval)

    # -------------------------------------------------- orblib + intake helpers
    def _orblib_parameters(self):
        """Parameters that define the orbit library.

        Same rule as ModelInnerIterator.__init__: everything except ml and
        the parameters of the chi2_ext component, since neither changes the
        orbits. Dropping only ml would treat an ext-only change as a new
        potential and re-integrate a library that could have been reused.
        """
        names = list(self.config.parspace.par_names)
        non_orblib = {"ml"}
        if getattr(self.config.system, "has_chi2_ext", False):
            ext = self.config.system.get_unique_ext_chi2_component()
            non_orblib.update(p.name for p in ext.parameters)
        return [n for n in names if n not in non_orblib]

    def _orblib_index(self, orblib_pars):
        """{potential -> orblib prefix} for every already-named row.

        Built once per _assign_directories instead of rescanning the table
        per row, which was quadratic in the number of models.
        """
        t = self.config.all_models.table
        cols = [c for c in orblib_pars if c in t.colnames]
        index = []
        for j in range(len(t)):
            d = str(t["directory"][j])
            if not d or d.startswith("rejected"):
                continue
            index.append((np.array([float(t[c][j]) for c in cols]),
                          d.rstrip("/").rsplit("/", 1)[0]))
        return cols, index

    def _existing_orblib_dir(self, idx, cols, index):
        """The orblib prefix of an earlier row with the same potential, or
        None. Same np.allclose match as ModelInnerIterator.is_new_orblib."""
        if not cols:
            return None
        t = self.config.all_models.table
        row = np.array([float(t[c][idx]) for c in cols])
        for potential, prefix in index:
            if np.allclose(row, potential):
                return prefix
        return None

    def _intake_bounds(self):
        bounds = {}
        for p in self.config.parspace:
            st = getattr(p, "par_generator_settings", None) or {}
            bounds[p.name] = {"lo": st.get("lo"), "hi": st.get("hi")}
        return bounds

    def _qobs(self):
        from dynamite.parameter_space import get_qobs_from_system

        return get_qobs_from_system(self.config.system)

    def _shape_param_names(self):
        """{'q': 'q-stars', ...} for the STARS component only.

        Resolved with the component's own get_parname rather than by
        splitting on '-': TriaxialCoredLogPotential declares p and q too, so
        a config with that halo has both p-stars and p-dh, and the
        triaxiality gate must not test the halo's axis ratios.
        """
        for cmp in self.config.system.cmp_list:
            if type(cmp).__name__ != "TriaxialVisibleComponent":
                continue
            names = {cmp.get_parname(p.name): p.name for p in cmp.parameters}
            missing = {"q", "p", "u"} - set(names)
            if missing:
                # validate() guarantees par == [q, p, u], so this cannot
                # happen -- but returning {} on a partial match would leave
                # validate_parset with no shape gate and no way to tell that
                # apart from a system that legitimately has no stars.
                raise ValueError(f"{cmp.name} lacks shape parameter(s) {sorted(missing)}")
            return {k: names[k] for k in ("q", "p", "u")}
        return {}

    def proposer_u_fixed(self, shape_names=None):
        """The stars component's u, when it is held fixed."""
        if shape_names is None:
            shape_names = self._shape_param_names()
        u_name = shape_names.get("u")
        for p in self.config.parspace:
            if p.name == u_name and getattr(p, "fixed", True):
                return float(p.raw_value)
        return None

# ------------------------------------------------------------------ helpers
def _read_weights_meta(model_full_dir):
    from astropy.io import ascii

    wfile = os.path.join(model_full_dir, WEIGHTS)
    meta = ascii.read(wfile).meta
    return {
        "chi2": float(meta["chi2_tot"]),
        "kinchi2": float(meta["chi2_kin"]),
        "kinmapchi2": float(meta["chi2_kinmap"]),
    }






def main(argv=None):
    ap = argparse.ArgumentParser(description="VERA evaluator driver")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="single step then exit")
    args = ap.parse_args(argv)

    import dynamite as dyn

    c = dyn.config_reader.Configuration(args.config, reset_logging=True)
    from .proposer_gridwalk import GridWalkProposer

    prop = GridWalkProposer(c)
    drv = VeraDriver(c, prop, run_dir=args.run_dir)
    drv.run_forever(dry_run=args.dry_run, once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
