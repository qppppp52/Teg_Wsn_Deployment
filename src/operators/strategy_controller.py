class StrategyController:
    def __init__(self, config):
        mc = config.get("mode", {})
        self.F = mc.get("F_init", 0.5)
        self.crossover_rate = mc.get("crossover_rate_init", 0.8)
        self.F_min = mc.get("F_min", 0.3)
        self.F_max = mc.get("F_max", 0.9)
        self.CR_min = mc.get("crossover_rate_min", 0.1)
        self.CR_max = mc.get("crossover_rate_max", 0.9)
        self.mutation_strategy = mc.get("mutation_strategy", "standard_rand")
        self.repair_strategy = mc.get("repair_strategy", "balanced")
