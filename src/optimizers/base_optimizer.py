"""Base optimizer state shared by all algorithm variants."""


class BaseOptimizer:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.config = config
        self.archive = None
        self.boost_invocations = 0
        self.boost_candidate_evaluations = 0
        self.boost_accepted_steps = 0
        self.boost_runtime_ms = 0.0

    def _record_boost_statistics(self, solutions):
        for solution in solutions:
            metadata = solution.metadata
            self.boost_invocations += int(metadata.get("boost_invocations", 0))
            self.boost_candidate_evaluations += int(
                metadata.get("boost_candidate_evaluations", 0)
            )
            self.boost_accepted_steps += int(metadata.get("boost_sinks_added", 0))
            self.boost_runtime_ms += float(metadata.get("boost_runtime_ms", 0.0))

    def run(self):
        raise NotImplementedError
