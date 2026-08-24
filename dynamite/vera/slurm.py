"""Thin, testable wrappers around the Slurm CLI (spec section 6).

Every public function takes a `runner` callable (argv -> stdout string) so
tests inject canned responses and the driver never shells out unmocked.
Job specifications are pinned to the values in the spec's Global Constraints.
"""

import getpass
import json
import os
import subprocess
import tempfile
import uuid

ACCOUNT = "mia"
SOLVE_SPEC = dict(partition="p.large", mem_mb=200000, cpus=24, time="06:00:00")
INT_SPEC = dict(partition="p.vera", mem_mb=None, cpus=72, time="08:00:00")
MAX_ARRAY_SIZE = 1001


class SlurmError(RuntimeError):
    pass


def _current_user():
    return os.environ.get("VERA_USER") or getpass.getuser()


def run_argv(argv):
    """Runs argv for real; stdout only. Non-zero exit raises SlurmError.

    A runner is just argv -> stdout, so this is a function; it held no state
    and every use site was `RealRunner()(argv)`.
    """
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SlurmError(f"{argv[0]} failed rc={proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def atomic_write(path, text):
    """Write `text` to `path` via a same-directory temp file and rename.

    Every persistent file vera writes lands on NFS where a reader can catch a
    partial flush -- the ledgers, the manifests, the parset files. One
    implementation so they cannot drift.
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=1))


def _base_flags(spec):
    flags = [
        f"--partition={spec['partition']}",
        f"--time={spec['time']}",
        f"--account={ACCOUNT}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={spec['cpus']}",
    ]
    if spec.get("mem_mb"):
        flags.append(f"--mem={spec['mem_mb']}")
    if spec.get("extra"):
        flags.extend(spec["extra"])
    return flags


def _array_flag(n_items, throttle):
    """`--array=0-N%k`, with N bounded by the scheduler's MaxArraySize.

    The throttle is a concurrency limit and cannot overflow; n_items is the
    one Slurm rejects the submission over.
    """
    n = max(0, min(int(n_items), MAX_ARRAY_SIZE) - 1)
    k = max(1, int(throttle))
    return f"--array=0-{n}%{k}"


def build_solve_job_spec(k, n_items):
    return {
        **SOLVE_SPEC,
        "extra": [_array_flag(n_items, k), "--job-name=ocen-solve"],
    }


def build_integration_job_spec(n_items):
    return {
        **INT_SPEC,
        "extra": [
            "--exclusive",
            _array_flag(n_items, 8),
            "--job-name=ocen-int",
        ],
    }


def write_manifest(manifest_dir, kind, items):
    """One item per line; the array task selects its own by index.

    Slurm hands every task of an array the SAME argv, so the per-task
    argument cannot be passed on the command line -- it has to be looked up
    from SLURM_ARRAY_TASK_ID. A file also keeps argv well clear of ARG_MAX
    for a thousand-model wave.
    """
    path = os.path.join(manifest_dir, f"vera_items_{kind}_{uuid.uuid4().hex[:8]}.txt")
    atomic_write(path, "".join(f"{item}\n" for item in items))
    return path


def submit_array(runner, job_spec, script_path, items, manifest_path):
    """`items` are per-task arguments, resolved by the task from the manifest."""
    argv = ["sbatch"] + _base_flags(job_spec) + [script_path, manifest_path]
    try:
        out = runner(argv)
    except Exception as e:
        # a runner may raise anything; callers catch SlurmError only
        raise e if isinstance(e, SlurmError) else SlurmError(
            f"sbatch invocation failed: {e!r}"
        ) from e
    for token in out.split():
        if token.isdigit():
            return int(token)
    raise SlurmError(f"unparseable sbatch output: {out!r}")


def running_job_ids(runner):
    # NOT "$USER": there is no shell here to expand it, and squeue would
    # either error or match nothing -- making every live job look dead.
    out = runner(["squeue", "-u", _current_user(), "-h", "-o", "%i"])
    ids = set()
    for line in out.splitlines():
        tok = line.strip().split("_")[0]
        if tok.isdigit():
            ids.add(int(tok))
    return ids


def levelfs(runner, user):
    # -P: without it sshare prints whitespace-aligned columns, the "|" split
    # below never yields 5 fields, and the adaptive throttle silently dies.
    out = runner(
        [
            "sshare",
            "-U",
            "-P",
            "--noheader",
            "--format=User,Account,RawUsage,NormUsage,LevelFS",
        ]
    )
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0] == user and len(parts) >= 5:
            try:
                return float(parts[4])
            except ValueError:
                return None
    return None
