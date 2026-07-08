from src.heatsink.sink_requirement import compute_sink_requirements
from src.heatsink.sink_allocator import allocate_all_sinks

def decode_heatsinks(solution, ctx):
    compute_sink_requirements(solution, ctx)
    allocate_all_sinks(solution, ctx)
