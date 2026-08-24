"""Shared base for the table-driven proposer adapters (spec section 4.1).

Every proposer wraps an existing ParameterGenerator. The generator appends
its parsets as rows of the shared all_models table, so observation flows
through the table and only failures need a channel of their own.
"""

import logging

from .proposal import Proposal, canonical_hash


class TableProposer:
    #: generator_type the config must name, and the parameter_space class
    GENERATOR_NAME = None

    def __init__(self, config):
        import dynamite.parameter_space as ps

        self.log = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config = config
        pss = config.settings.parameter_space_settings
        gen_type = pss["generator_type"]
        if gen_type != self.GENERATOR_NAME:
            raise ValueError(
                f"{type(self).__name__} needs generator_type "
                f"{self.GENERATOR_NAME}, got {gen_type}"
            )
        self.generator = getattr(ps, self.GENERATOR_NAME)(
            config.parspace, parspace_settings=pss
        )
        self.par_names = list(config.parspace.par_names)
        self.pid_to_row = {}  # proposal_id -> table row index
        self.failed_pids = set()  # intake rejections + parked models

    def _row_to_parset(self, row_idx):
        t = self.config.all_models.table
        return {name: float(t[name][row_idx]) for name in self.par_names}

    def propose(self, max_batch=None):
        """Every row the generator appended becomes a proposal.

        max_batch is advisory: a row left un-proposed gets no proposal_id and
        no directory, is skipped by the driver's scan, and is never revisited
        -- the parameter set would vanish. Batch size is a generator setting.
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
            pid = canonical_hash(parset)
            self.pid_to_row[pid] = i
            props.append(Proposal(proposal_id=pid, parset=parset))
            t["directory"][i] = t["directory"][i] or f"pending/{pid}"
        self.log.info("propose(): %d new proposal(s)", len(props))
        return props

    def observe(self, results):
        """Chi2 reaches the generator through the table; failures never appear
        there at all, so they are recorded here."""
        for r in results:
            if getattr(r, "status", None) == "failed":
                self.failed_pids.add(r.proposal_id)

    def quorum_pending(self):
        t = self.config.all_models.table
        return sum(
            1 for row in self.pid_to_row.values() if not bool(t["all_done"][row])
        )

    def exhausted(self):
        raise NotImplementedError
