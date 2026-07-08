"""散热片占位分配 — 按能量需求排序分配，减少争用"""
from src.heatsink.sink_overlap_rules import SinkOverlapRules


def _allocate_for_node(node_id, required, solution, ctx, rules):
    """为单个节点贪婪分配散热片网格"""
    a = []
    for n in ctx.neighbor_sets[node_id]:
        if len(a) >= required:
            break
        if rules.can_place_sink_on_grid(n, node_id, solution, ctx):
            a.append(int(n))
    return a


def allocate_all_sinks(solution, ctx):
    """全局散热片分配：先清空再按能量需求排序分配"""
    rules = SinkOverlapRules(ctx.config)

    # 清空已有的散热片占位
    for i in range(ctx.num_candidates):
        solution.z_sink_sensor[i] = []
        solution.z_sink_ap[i] = []

    # 收集所有需要散热片的节点，按n_sink需求降序排序
    sensor_nodes = [(si, solution.n_sink_sensor[si])
                     for si in range(ctx.num_candidates)
                     if solution.x[si] == 1 and solution.n_sink_sensor[si] > 0]
    ap_nodes = [(ai, solution.n_sink_ap[ai])
                 for ai in range(ctx.num_candidates)
                 if solution.y[ai] == 1 and solution.n_sink_ap[ai] > 0]

    # 需求大的先分配，减少争用
    sensor_nodes.sort(key=lambda t: t[1], reverse=True)
    ap_nodes.sort(key=lambda t: t[1], reverse=True)

    for si, req in sensor_nodes:
        solution.z_sink_sensor[si] = _allocate_for_node(si, req, solution, ctx, rules)

    for ai, req in ap_nodes:
        solution.z_sink_ap[ai] = _allocate_for_node(ai, req, solution, ctx, rules)
