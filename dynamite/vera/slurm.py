"""Thin, testable wrappers around the Slurm CLI (spec section 6).

Every public function takes a `runner` callable (argv -> stdout string) so
tests inject canned responses and the driver never shells out unmocked.
Job specifications are pinned to the values in the spec's Global Constraints.
"""

import getpass
import os
import subprocess

ACCOUNT = "mia"
SOLVE_SPEC = dict(partition="p.large", mem_mb=200000, cpus=24, time="06:00:00")
INT_SPEC = dict(partition="p.vera", mem_mb=None, cpus=72, time="08:00:00")
MAX_ARRAY_SIZE = 1001


class SlurmError(RuntimeError):
    pass


def _current_user():
    return os.environ.get("VERA_USER") or getpass.getuser()


class RealRunner:
    """Runs argv for real; stdout only. Non-zero exit raises SlurmError."""

    def __call__(self, argv):
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise SlurmError(
                f"{argv[0]} failed rc={proc.returncode}: {proc.stderr.strip()}"
            )
        return proc.stdout


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
    k = max(1, min(int(throttle), MAX_ARRAY_SIZE))
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


def submit_array(runner, job_spec, script_path, items):
    """items are per-task argument strings joined with ';' into one array."""
    argv = ["sbatch"] + _base_flags(job_spec) + [script_path, ";".join(items)]
    try:
        out = runner(argv)
    except SlurmError:
        raise
    except Exception as e:
        raise SlurmError(f"sbatch invocation failed: {e!r}") from e
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
    out = runner(
        [
            "sshare",
            "-U",
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
