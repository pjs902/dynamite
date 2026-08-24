"""TableDriven GridWalk adapter (spec sections 4 and 4.1)."""

from .proposer import TableProposer


class GridWalkProposer(TableProposer):
    GENERATOR_NAME = "GridWalk"

    def exhausted(self):
        # the generator's own criteria first -- counting only completed models
        # left a converged campaign (generate() returning nothing) looking
        # unfinished forever, idling the allocation
        if super().exhausted():
            return True
        # ...plus the async criterion the generator cannot see: it counts rows,
        # we wait for the models behind them to actually finish
        stop = self.config.settings.parameter_space_settings["stopping_criteria"]
        t = self.config.all_models.table
        return sum(1 for r in t if bool(r["all_done"])) >= stop["n_max_mods"]
