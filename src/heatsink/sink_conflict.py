"""散热片冲突检查与空间不重叠约束（建模文件公式）"""


def check_sink_conflicts(solution, ctx):
    """
    检查散热片间重叠冲突。对任意网格，若有 >1 个散热片声称拥有 → 冲突。
    """
    all_sinks = {}
    for si in range(ctx.num_candidates):
        for g in solution.z_sink_sensor[si]:
            all_sinks.setdefault(g, []).append(f"s{si}")
        for g in solution.z_sink_ap[si]:
            all_sinks.setdefault(g, []).append(f"a{si}")
    conflicts = 0
    for owners in all_sinks.values():
        if len(owners) > 1:
            conflicts += len(owners) - 1
    return conflicts


def check_spatial_non_overlap(solution, ctx):
    """
    建模文件公式：
      对于任意网格 k，其至多被一个物理对象占用。
      允许散热片与其自身节点共占同一格点（x_k=1 和 z_kk=1 可并存），
      除此之外禁止任何重叠。

      x_k + y_k + Σ_{i≠k} z_ik^sensor + Σ_{j≠k} z_jk^ap ≤ 1

    返回违反次数。
    """
    K = ctx.num_candidates
    violations = 0

    for k in range(K):
        occupants = []

        # 网格 k 上的部署节点
        if solution.x[k] == 1:
            occupants.append(("node", k))
        if solution.y[k] == 1:
            occupants.append(("node", k))

        # 网格 k 上的散热片
        for i in range(K):
            if k in solution.z_sink_sensor[i]:
                occupants.append(("sink", i))
            if k in solution.z_sink_ap[i]:
                occupants.append(("sink", i))

        if len(occupants) <= 1:
            continue

        # 合法自重叠检查：必须是 节点(k)+自身散热片(k)
        if len(occupants) == 2:
            types = {o[0] for o in occupants}
            ids = {o[1] for o in occupants}
            node_types = {"node"}
            sink_types = {"sink"}
            if types.intersection(node_types) and types.intersection(sink_types) and len(ids) == 1:
                # 自身节点 + 自身散热片 → 允许
                continue

        # 其他所有情况 → 违反
        violations += len(occupants)

    return violations
