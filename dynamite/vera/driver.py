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
import os
import sys
import tempfile
import time

import numpy as np

from .classifier import ModelState, classify
from .pack_integration import pack_libraries
from .proposal import Result
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


class VeraDriver:
    def __init__(
        self,
        config,
        proposer,
        runner=None,
        run_dir=".",
        poll_interval=POLL_INTERVAL_S,
        k_start=K_START,
    ):
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
        self.dir_to_pid = {}
        self.dir_to_row = {}

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
        now = time.time()
        for row in self.config.all_models.table:
            d = row["directory"]
            if not d:
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

    def step(self, dry_run=False):
        self.observe_completions()
        self.reconcile_and_submit(dry_run=dry_run)
        if self.proposer.quorum_pending() == 0 and not self.proposer.exhausted():
            for p in self.proposer.propose():
                self._create_model_entry(p)
            self.reconcile_and_submit(dry_run=dry_run)
        return not self.proposer.exhausted()

    def run_forever(self, dry_run=False, once=False):
        while True:
            alive = self.step(dry_run=dry_run)
            if once or not alive:
                break
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------ model rows
    def _create_model_entry(self, proposal):
        """Append one table row per proposal; assign a model directory.

        Directory scheme mirrors ModelInnerIterator.assign_model_directories:
        orblib_<iter>_<seq>/ml<value>. The proposer records dir->pid so
        observe_completions can attribute results.
        """
        t = self.config.all_models.table
        seq = sum(1 for r in t if str(r["directory"]).startswith("orblib_"))
        which_iter = int(max(t["which_iter"])) + 1 if len(t) else 0
        ml = proposal.parset.get("ml")
        ml_tag = f"ml{ml:.2f}" if ml is not None else "ml0.00"
        directory = f"orblib_{which_iter:03d}_{seq:03d}/{ml_tag}"
        t.add_row(
            [
                *[proposal.parset.get(p.name, np.nan) for p in self.config.parspace],
                np.nan,
                np.nan,
                np.nan,  # chi2 columns
                str(_now64()),  # time_modified
                False,
                False,
                False,  # flags
                which_iter,
                directory,
            ]
        )
        row = len(t) - 1
        self.dir_to_pid[directory] = proposal.proposal_id
        self.dir_to_row[directory] = row
        self._save_table_atomically()


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
