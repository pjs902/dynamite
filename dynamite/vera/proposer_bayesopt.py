"""TableDriven adapter around BayesOptGenerator (spec sections 4.1, C3).

BayesOptGenerator trains its GP from the shared all_models table and
appends proposals as rows, so observation flows through the table and
``observe()`` is a no-op. Quorum is continuous: quorum_pending() returns 0
whenever there is nothing unsolved we are waiting on; stopping ORs the
generator's status flags including the R3 ``gp_predictions_accurate``.
"""

import logging

from .proposal import Proposal


class BayesOptProposer:
    STATUS_FLAGS = (
        "stop",
        "n_max_mods_reached",
        "n_max_iter_reached",
        "gp_max_variance_low",
        "gp_min_ei_low",
        "gp_predictions_accurate",
    )

    def __init__(self, config):
        import dynamite.parameter_space as ps

        self.log = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config = config
        pss = config.settings.parameter_space_settings
        gen_type = pss["generator_type"]
        if gen_type != "BayesOpt":
            raise ValueError(
                f"BayesOptProposer needs generator_type BayesOpt, got {gen_type}"
            )
        self.generator = ps.BayesOptGenerator(config.parspace, parspace_settings=pss)
        self.par_names = [p.name for p in config.parspace]
        self.pid_to_row = {}
        self.failed_pids = set()  # intake rejections + parked models

    def _row_to_parset(self, row_idx):
        t = self.config.all_models.table
        return {name: float(t[name][row_idx]) for name in self.par_names}

    def propose(self, max_batch=4):
        t = self.config.all_models.table
        before = len(t)
        self.generator.generate(current_models=self.config.all_models)
        props = []
        for i in range(before, len(t)):
            parset = self._row_to_parset(i)
            from .proposal import canonical_hash

            pid = canonical_hash(parset)
            self.pid_to_row[pid] = i
            props.append(Proposal(proposal_id=pid, parset=parset))
            if len(props) >= max_batch:
                break
        self.log.info("propose(): %d new proposal(s)", len(props))
        return props

    def observe(self, results):
        """Chi2 flows through the table at the next generate(); failures do
        not appear there at all, so they are recorded here.
        """
        for r in results:
            if getattr(r, "status", None) == "failed":
                self.failed_pids.add(r.proposal_id)
        return None

    def tracked_results(self):
        return dict(self.pid_to_row)

    def quorum_pending(self):
        t = self.config.all_models.table
        return sum(
            1 for row in self.pid_to_row.values() if not bool(t["all_done"][row])
        )

    def exhausted(self):
        status = getattr(self.generator, "status", {}) or {}
        return any(bool(status.get(f)) for f in self.STATUS_FLAGS)
