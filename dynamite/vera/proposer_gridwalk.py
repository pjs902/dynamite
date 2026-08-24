"""TableDriven GridWalk adapter (spec sections 4 and 4.1).

Wraps the existing ``GridWalk`` ParameterGenerator so the driver can ask a
dynamite-native walker for proposals without ModelIterator. Observation
flows through the shared ``all_models`` table - ``observe()`` is therefore
a no-op; ``quorum_pending()``/``exhausted()`` read the table.
"""

import logging

from .proposal import Proposal


class GridWalkProposer:
    def __init__(self, config):
        import dynamite.parameter_space as ps

        self.log = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config = config
        pss = config.settings.parameter_space_settings
        gen_type = pss["generator_type"]
        if gen_type != "GridWalk":
            raise ValueError(
                f"GridWalkProposer needs generator_type GridWalk, got {gen_type}"
            )
        self.generator = ps.GridWalk(config.parspace, parspace_settings=pss)
        self.par_names = [p.name for p in config.parspace]
        self.pid_to_row = {}  # proposal_id -> table row index
        self.failed_pids = set()  # intake rejections + parked models

    # ------------------------------------------------------------------
    def _row_to_parset(self, row_idx):
        t = self.config.all_models.table
        return {name: float(t[name][row_idx]) for name in self.par_names}

    def propose(self, max_batch=None):
        """Every row the generator appended becomes a proposal.

        max_batch is advisory only: a row left un-proposed here gets no
        proposal_id and no directory, is skipped by the driver's scan, and
        is never revisited -- the parameter set would simply vanish from
        the campaign. Batch size belongs to the generator settings.
        """
        t = self.config.all_models.table
        before = len(t)
        self.generator.generate(current_models=self.config.all_models)
        if max_batch is not None and len(t) - before > max_batch:
            self.log.warning(
                "generator produced %d rows, above the advisory max_batch %d; "
                "proposing all of them", len(t) - before, max_batch,
            )
        props = []
        for i in range(before, len(t)):
            parset = self._row_to_parset(i)
            pid = canonical_hash_stable(parset)
            self.pid_to_row[pid] = i
            props.append(Proposal(proposal_id=pid, parset=parset))
            t["directory"][i] = t["directory"][i] or f"pending/{pid}"
        self.log.info("propose(): %d new proposal(s)", len(props))
        return props

    def observe(self, results):
        """The table carries the chi2 values, but failures never reach it:
        a rejected or parked proposal has no row worth reading, so record
        it here rather than dropping it on the floor.
        """
        for r in results:
            if getattr(r, "status", None) == "failed":
                self.failed_pids.add(r.proposal_id)
        return None

    def tracked_results(self):
        """proposal_id -> row index map for outstanding/tracked proposals."""
        return dict(self.pid_to_row)

    def quorum_pending(self):
        if not self.pid_to_row:
            return 0
        t = self.config.all_models.table
        unsolved = sum(
            1 for pid, row in self.pid_to_row.items() if not bool(t["all_done"][row])
        )
        return unsolved

    def exhausted(self):
        stop = self.config.settings.parameter_space_settings["stopping_criteria"]
        t = self.config.all_models.table
        n_done = sum(1 for r in t if bool(r["all_done"]))
        return bool(n_done >= stop["n_max_mods"])


def canonical_hash_stable(parset):
    from .proposal import canonical_hash

    return canonical_hash(parset)
