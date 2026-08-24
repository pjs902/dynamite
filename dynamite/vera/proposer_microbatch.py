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

    def ready_to_propose(self):
        """Re-center as soon as `min_solved_fraction` of the batch is in --
        overriding the gate, not the count, so quorum_pending() stays honest.
        """
        tracked_n = len(self.pid_to_row)
        if tracked_n == 0:
            return True
        solved_frac = 1.0 - self.quorum_pending() / tracked_n
        return solved_frac >= self.min_solved_fraction
