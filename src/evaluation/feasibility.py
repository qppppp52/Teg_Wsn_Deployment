import numpy as np

def compute_feasibility_stats(solutions):
    if not solutions: return {"FR":0, "CV_avg":float("inf")}
    feasible = [s for s in solutions if s.feasible]
    FR = len(feasible)/len(solutions)
    CVs = [s.cv for s in solutions]
    return {"FR":FR, "CV_avg":float(np.mean(CVs))}
