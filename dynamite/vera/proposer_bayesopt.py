"""TableDriven adapter around BayesOptGenerator (spec sections 4.1, C3).

Stopping ORs the generator's own status flags, including the R3
``gp_predictions_accurate`` signal.
"""

from .proposer import TableProposer


class BayesOptProposer(TableProposer):
    GENERATOR_NAME = "BayesOptGenerator"

    STATUS_FLAGS = (
        "stop",
        "n_max_mods_reached",
        "n_max_iter_reached",
        "gp_max_variance_low",
        "gp_min_ei_low",
        "gp_predictions_accurate",
    )

    def exhausted(self):
        status = getattr(self.generator, "status", {}) or {}
        return any(bool(status.get(f)) for f in self.STATUS_FLAGS)
