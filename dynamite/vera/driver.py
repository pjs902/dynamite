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

from .classifier import ModelState, classify
from .pack_integration import pack_libraries
from .proposal import Result, validate_parset, SCHEMA_VERSION
from .slurm import (
    RealRunner,
    build_integration_job_spec,
    build_solve_job_spec,
    levelfs,
    running_job_ids,
    submit_array,
)

POLL_INTERVAL_S = 300
K_START = 16
USER = "pesmith"
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
        self.runner = runner or RealRunner()
        self.run_dir = os.path.abspath(run_dir)
        self.poll_interval = poll_interval
        self.k_start = k_start
        self.attempts = self._load_json(ATTEMPTS_FILE, {})
        self.inflight = self._load_json(LEDGER_FILE, {"int": [], "solve": []})
        self.output_root = config.settings.io_settings["output_directory"]
        # proposal attribution, owned by the driver (not the proposer):
        dirs_map = self._load_json(DIRS_FILE, {})
        self.dir_to_pid = dirs_map.get("dir_to_pid", {})
        self.dir_to_row = {}
        self.rejected = {}
        self.log = logging.getLogger(f"{__name__}.VeraDriver")

    # ------------------------------------------------------------ persistence
    def _load_json(self, name, default):
        p = os.path.join(self.run_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        return default

    def _dump_json(self, name, obj):
        fd, tmp = tempfile.mkstemp(dir=self.run_dir)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, os.path.join(self.run_dir, name))

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

    def reconcile_and_submit(self, dry_run=False):
        states = self.scan()
        live = set() if dry_run else running_job_ids(self.runner)
        for kind in ("int", "solve"):
            live_items = []
            for item in self.inflight.get(kind, []):
                jid = _ledger_jid(self.run_dir, kind, item)
                if jid is None or jid in live:
                    live_items.append(item)
            self.inflight[kind] = live_items

        to_int = sorted(
            d for d, s in states.items() if s is ModelState.PENDING_INTEGRATION
        )
        to_sol = sorted(d for d, s in states.items() if s is ModelState.TO_SOLVE)

        for d in list(to_int) + list(to_sol):
            self._write_parset_file(d)
        if os.environ.get("VERA_DEBUG"):
            print(f"[dbg] states={ {k: v.value for k, v in states.items()} } "
                  f"to_int={to_int} to_sol={to_sol} "
                  f"inflight={self.inflight}", flush=True)
        submitted = 0
        if to_int:
            submitted += self._submit_wave("int", pack_libraries(to_int), dry_run)
        if to_sol:
            # a built library always earns its weights: chi2 feeds every
            # future proposer decision. The quorum gate lives on PROPOSING
            # new work (step()), never on draining existing artifacts.
            k = self._adaptive_k()
            submitted += self._submit_wave("solve", [[d] for d in to_sol], dry_run, k=k)
        return submitted

    def _submit_wave(self, kind, packages, dry_run, k=None):
        done = set(self.inflight.get(kind, []))
        if kind == "int":
            items = [";".join(pkg) for pkg in packages]
        else:
            items = [pkg[0] for pkg in packages]
        new = [it for it in items if it not in done]
        if not new:
            return 0
        if dry_run:
            print(
                f"[dry-run] would submit {len(new)} {kind} package(s): "
                f"{new[:3]}{'...' if len(new) > 3 else ''}"
            )
            return 0
        if kind == "int":
            spec = build_integration_job_spec(n_items=len(new))
            script = os.path.join(
                os.path.dirname(__file__), "scripts", "integrate_package.sh"
            )
        else:
            spec = build_solve_job_spec(k=k or self.k_start, n_items=len(new))
            script = os.path.join(os.path.dirname(__file__), "scripts", "solve_task.sh")
        if hasattr(self.runner, "submit_array"):
            # local backend: execute synchronously, skip slurm argv plumbing
            jid = self.runner.submit_array(script, items)
        else:
            jid = submit_array(self.runner, spec, script, items)
        for i, item in enumerate(items):
            _remember_ledger_jid(self.run_dir, kind, item, jid + i)
        self.inflight.setdefault(kind, []).extend(new)
        self._dump_json(LEDGER_FILE, self.inflight)
        print(f"submitted {kind} array job {jid} with {len(new)} task(s)")
        return 1

    def _adaptive_k(self):
        lf = levelfs(self.runner, USER)
        if lf is None:
            return self.k_start
        if lf > 10.0:
            return min(24, self.k_start + 4)
        if lf < 1.0:
            return max(4, self.k_start - 4)
        return self.k_start


    def _write_parset_file(self, model_dir):
        """Drop <models>/<dir>/vera_parset.json next to the artifacts.

        Workers are pure functions of (config, parset): they never read the
        shared all_models table, because every Configuration init runs
        update_model_table() - whose janitor would delete *other* pending
        rows it finds there (concurrent readers are outside DYNAMITE's
        contract). Full par-values, fixed parameters included.
        """
        t = self.config.all_models.table
        match = [i for i, r in enumerate(t) if r["directory"] == model_dir]
        if not match:
            return
        i = match[0]
        payload = {"schema_version": SCHEMA_VERSION,
                   "par_names": list(self.config.parspace.par_names),
                   "values": {n: float(t[n][i])
                              for n in self.config.parspace.par_names}}
        target = os.path.join(self.output_root, "models", model_dir,
                              "vera_parset.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target))
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, target)

    def observe_completions(self):
        states = self.scan()
        results = []
        t = self.config.all_models.table
        for d, state in states.items():
            pid = self.dir_to_pid.get(d)
            if pid is None:
                continue
            row = self.dir_to_row[d]
            if state is ModelState.SOLVED and not bool(t["all_done"][row]):
                meta = _read_weights_meta(os.path.join(self.output_root, "models", d))
                results.append(
                    Result(proposal_id=pid, model_dir=d, status="done", **meta)
                )
                t["chi2"][row] = meta["chi2"]
                t["kinchi2"][row] = meta["kinchi2"]
                t["kinmapchi2"][row] = meta["kinmapchi2"]
                t["all_done"][row] = True
                t["weights_done"][row] = True
                t["time_modified"][row] = str(_now64())
        failed = [d for d, s in states.items() if s is ModelState.PARKED]
        for d in failed:
            pid = self.proposer.dir_to_pid.get(d)
            if pid is not None:
                results.append(Result(proposal_id=pid, model_dir=d, status="failed"))
        if results:
            self.proposer.observe(results)
            self._save_table_atomically()
        return len(results)

    def exhausted(self):
        """Campaign finished according to the proposer's stopping rules."""
        return self.proposer.exhausted()

    def step(self, dry_run=False):
        self.observe_completions()
        self.reconcile_and_submit(dry_run=dry_run)
        if self.proposer.quorum_pending() == 0 and not self.proposer.exhausted():
            t = self.config.all_models.table
            before = len(t)
            self.proposer.propose()
            self._assign_directories(range(before, len(t)))
            self.reconcile_and_submit(dry_run=dry_run)
        return not self.proposer.exhausted()

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
        u_fixed = self.proposer_u_fixed()
        sformat = self.config.system.parameters[0].sformat

        row_to_pid = {row: pid for pid, row in
                      self.proposer.pid_to_row.items()}

        for idx in row_indices:
            pid = row_to_pid.get(idx)
            if pid is None:
                continue
            parset = {name: float(t[name][idx])
                      for name in self.proposer.par_names}
            clipped, violations = validate_parset(
                parset, bounds, qobs=qobs, u_fixed=u_fixed)
            if violations:
                self.log.warning("intake rejected %s: %s", pid, violations)
                self.rejected[pid] = violations
                t["directory"][idx] = f"rejected/{pid}/"
                result = Result(proposal_id=pid,
                                model_dir=t["directory"][idx],
                                status="failed")
                self.proposer.observe([result])
                del self.proposer.pid_to_row[pid]
                continue
            iteration = int(t["which_iter"][idx])
            prior = t[:idx]
            n = sum(1 for r in prior
                    if int(r["which_iter"]) == iteration
                    and r["directory"]
                    and not str(r["directory"]).startswith("rejected"))
            ml_val = float(t["ml"][idx]) if "ml" in t.colnames else 0.0
            directory = (f"orblib_{iteration:03d}_{n:03d}"
                         f"/ml{ml_val:{sformat}}/")
            t["directory"][idx] = directory
            self.dir_to_pid[directory] = pid
            self.dir_to_row[directory] = idx
        self._save_table_atomically()
        self._dump_json(DIRS_FILE, {"dir_to_pid": self.dir_to_pid})

    def run_forever(self, dry_run=False, once=False):
        while True:
            alive = self.step(dry_run=dry_run)
            if once or not alive:
                break
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------ model rows
    def _intake_bounds(self):
        bounds = {}
        for p in self.config.parspace:
            st = getattr(p, "par_generator_settings", None) or {}
            bounds[p.name] = {"lo": st.get("lo"), "hi": st.get("hi")}
        return bounds

    def _qobs(self):
        try:
            stars = [c for c in self.config.system.cmp_list
                     if type(c).__name__ == "TriaxialVisibleComponent"]
            return float(stars[0].qobs) if stars else None
        except (AttributeError, IndexError, TypeError):
            return None

    def _intake_bounds(self):
        bounds = {}
        for p in self.config.parspace:
            st = getattr(p, "par_generator_settings", None) or {}
            bounds[p.name] = {"lo": st.get("lo"), "hi": st.get("hi")}
        return bounds

    def _qobs(self):
        try:
            stars = [c for c in self.config.system.cmp_list
                     if type(c).__name__ == "TriaxialVisibleComponent"]
            return float(stars[0].qobs) if stars else None
        except (AttributeError, IndexError, TypeError):
            return None

    def proposer_u_fixed(self):
        for p in self.config.parspace:
            if p.name == "u" and getattr(p, "fixed", True):
                return float(p.raw_value)
        return None

# ------------------------------------------------------------------ helpers
def _now64():
    import numpy as np

    return np.datetime64("now", "ms")


def _read_weights_meta(model_full_dir):
    from astropy.io import ascii

    wfile = os.path.join(model_full_dir, "orbit_weights.ecsv")
    meta = ascii.read(wfile).meta
    return {
        "chi2": float(meta["chi2_tot"]),
        "kinchi2": float(meta["chi2_kin"]),
        "kinmapchi2": float(meta["chi2_kinmap"]),
    }


def _ledger_path(run_dir):
    return os.path.join(run_dir, "vera_ledger_jids.json")


def _remember_ledger_jid(run_dir, kind, item, jid):
    p = _ledger_path(run_dir)
    data = {}
    if os.path.isfile(p):
        with open(p) as f:
            data = json.load(f)
    data[f"{kind}:{item}"] = jid
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)


def _ledger_jid(run_dir, kind, item):
    p = _ledger_path(run_dir)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f).get(f"{kind}:{item}")


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
