"""TableDriven adapter around BayesOptGenerator (spec sections 4.1, C3).

Stopping ORs the generator's own status flags, including the R3
``gp_predictions_accurate`` signal.
"""

from .proposer import TableProposer


class BayesOptProposer(TableProposer):
    GENERATOR_NAME = "BayesOptGenerator"

    # stopping comes from TableProposer.exhausted(): status["stop"] already ORs
    # the GP flags (gp_max_variance_low, gp_min_ei_low, gp_predictions_accurate)
    # along with every generic criterion.
