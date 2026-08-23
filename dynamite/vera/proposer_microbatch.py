"""Micro-batch walker: re-center on a solved fraction, not full batches
(spec section 10, phase C2).

Inherits the GridWalk machinery; only the quorum semantics change. When the
solved fraction of tracked proposals reaches ``min_solved_fraction``, the
driver may immediately ask for a fresh, re-centered batch.
"""

from .proposer_gridwalk import GridWalkProposer


class MicroBatchWalkProposer(GridWalkProposer):
    def __init__(self, config, min_solved_fraction=0.8):
        super().__init__(config)
        if not 0.0 < float(min_solved_fraction) <= 1.0:
            raise ValueError("min_solved_fraction must be in (0, 1]")
        self.min_solved_fraction = float(min_solved_fraction)

    def _outstanding(self):
        t = self.config.all_models.table
        return sum(
            1 for row in self.pid_to_row.values() if not bool(t["all_done"][row])
        )

    def quorum_pending(self):
        tracked_n = len(self.pid_to_row)
        if tracked_n == 0:
            return 0
        remaining = self._outstanding()
        solved_frac = 1.0 - remaining / tracked_n
        return 0 if solved_frac >= self.min_solved_fraction else remaining

    def exhausted(self):
        # stopping still governed by n_max_mods via the parent
        return super().exhausted()
