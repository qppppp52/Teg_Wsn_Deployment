class BaseOptimizer:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.config = config
        self.archive = None
    def run(self):
        raise NotImplementedError
