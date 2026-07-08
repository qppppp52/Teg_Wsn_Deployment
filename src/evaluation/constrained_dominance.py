def constrained_dominates(a, b):
    if a.feasible and not b.feasible: return True
    if not a.feasible and b.feasible: return False
    if not a.feasible and not b.feasible: return a.cv < b.cv
    return (a.coverage>=b.coverage and a.throughput>=b.throughput and
            (a.coverage>b.coverage or a.throughput>b.throughput))
