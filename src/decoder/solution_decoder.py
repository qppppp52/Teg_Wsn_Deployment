from src.model.solution import Solution
from src.decoder.deployment_decoder import decode_deployment
from src.decoder.connection_decoder import assign_connections
from src.decoder.power_decoder import decode_power
from src.decoder.heatsink_decoder import decode_heatsinks

def decode_solution(individual, ctx):
    sol = Solution(ctx.num_candidates)
    sol.x, sol.y = decode_deployment(individual, ctx)
    sol.c = assign_connections(sol.x, sol.y, ctx)
    decode_power(sol, ctx)
    decode_heatsinks(sol, ctx)
    return sol
