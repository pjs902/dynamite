"""Local execution backend: run task scripts as plain subprocesses.

Lets the whole evaluator loop run on one machine with no Slurm - used by
the local end-to-end test and as an overflow/dev mode. Jobs execute
synchronously inside submit_array (fine: local tasks are small) and report
synthetic negative job ids so they are never mistaken for live Slurm jobs.
"""

import os
import subprocess

from .slurm import SlurmError, write_manifest


class LocalRunner:
    def __init__(self, env=None):
        self.env = env or {}
        self._next_jid = -1000
        self.executed = []  # (script, item) history for assertions

    def __call__(self, argv):
        """Query commands only; submissions go through submit_array()."""
        cmd = argv[0]
        if cmd == "squeue":
            return ""  # nothing persistent is ever live
        if cmd == "sshare":
            return "local|local|0|0|9999\n"
        raise SlurmError(f"LocalRunner cannot answer {cmd}")

    def submit_array(self, script_path, items):
        # Drive the scripts exactly as Slurm does -- one manifest for the
        # whole array, each task selecting its line by SLURM_ARRAY_TASK_ID.
        # Passing the item on the command line here would make this backend
        # more capable than the real one and hide indexing bugs.
        run_dir = self.env.get("VERA_RUN_DIR") or os.getcwd()
        manifest = write_manifest(run_dir, "local", items)
        for idx, item in enumerate(items):
            self.executed.append((script_path, item))
            proc = subprocess.run(
                ["bash", script_path, manifest],
                capture_output=True,
                text=True,
                env={**os.environ, **self.env, "SLURM_ARRAY_TASK_ID": str(idx)},
            )
            if proc.returncode != 0:
                tail = proc.stdout[-1500:] + proc.stderr[-1500:]
                raise SlurmError(
                    f"local task failed rc={proc.returncode}: {item}\n{tail}"
                )
        jid = self._next_jid
        self._next_jid -= len(items)
        return jid
