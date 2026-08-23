"""Local execution backend: run task scripts as plain subprocesses.

Lets the whole evaluator loop run on one machine with no Slurm - used by
the local end-to-end test and as an overflow/dev mode. Jobs execute
synchronously inside submit_array (fine: local tasks are small) and report
synthetic negative job ids so they are never mistaken for live Slurm jobs.
"""

import os
import subprocess

from .slurm import SlurmError


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
        for item in items:
            self.executed.append((script_path, item))
            proc = subprocess.run(
                ["bash", script_path, item],
                capture_output=True,
                text=True,
                env={**os.environ, **self.env},
            )
            if proc.returncode != 0:
                tail = proc.stdout[-1500:] + proc.stderr[-1500:]
                raise SlurmError(
                    f"local task failed rc={proc.returncode}: {item}\n{tail}"
                )
        jid = self._next_jid
        self._next_jid -= len(items)
        return jid
