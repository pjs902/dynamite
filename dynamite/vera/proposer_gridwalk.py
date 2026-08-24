"""TableDriven GridWalk adapter (spec sections 4 and 4.1)."""

from .proposer import TableProposer


class GridWalkProposer(TableProposer):
    GENERATOR_NAME = "GridWalk"

    def exhausted(self):
        stop = self.config.settings.parameter_space_settings["stopping_criteria"]
        t = self.config.all_models.table
        n_done = sum(1 for r in t if bool(r["all_done"]))
        return bool(n_done >= stop["n_max_mods"])
