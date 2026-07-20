"""Discrete generation-level action space for DQN-CR-MODE."""

REPAIR_STEPS = ("deploy", "link", "energy", "service", "sink")

ACTIONS = [
    {"id": 0, "name": "balanced", "F": 0.5, "CR": 0.8, "mutation_strategy": "rand_1", "repair_order": ["deploy", "link", "energy", "service", "sink"]},
    {"id": 1, "name": "exploration_high_F", "F": 0.8, "CR": 0.9, "mutation_strategy": "rand_1", "repair_order": ["deploy", "link", "energy", "service", "sink"]},
    {"id": 2, "name": "exploitation_low_F", "F": 0.35, "CR": 0.6, "mutation_strategy": "current_to_best_1", "repair_order": ["deploy", "link", "energy", "service", "sink"]},
    {"id": 3, "name": "energy_first", "F": 0.45, "CR": 0.7, "mutation_strategy": "rand_1", "repair_order": ["energy", "sink", "link", "service", "deploy"]},
    {"id": 4, "name": "link_first", "F": 0.55, "CR": 0.8, "mutation_strategy": "rand_1", "repair_order": ["link", "energy", "sink", "service", "deploy"]},
    {"id": 5, "name": "sink_first", "F": 0.45, "CR": 0.65, "mutation_strategy": "rand_1", "repair_order": ["sink", "energy", "link", "service", "deploy"]},
    {"id": 6, "name": "rsum_search_priority", "F": 0.6, "CR": 0.85, "mutation_strategy": "best_1", "repair_order": ["link", "energy", "sink", "service", "deploy"]},
    {"id": 7, "name": "diversity_boost", "F": 0.9, "CR": 0.95, "mutation_strategy": "rand_2", "repair_order": ["deploy", "link", "energy", "service", "sink"]},
    {"id": 8, "name": "conservative_repair", "F": 0.3, "CR": 0.5, "mutation_strategy": "current_to_best_1", "repair_order": ["energy", "sink", "link", "service", "deploy"]},
]


def get_action(action_id):
    action_id = int(action_id)
    if action_id < 0 or action_id >= len(ACTIONS):
        raise IndexError(f"DQN action_id {action_id} is outside [0, {len(ACTIONS) - 1}]")
    return ACTIONS[action_id]


def action_dim():
    return len(ACTIONS)


def effective_action_signature(action):
    return (
        float(action["F"]),
        float(action["CR"]),
        str(action["mutation_strategy"]),
        tuple(action["repair_order"]),
    )


def audit_action_space():
    by_signature = {}
    rows = []
    for action in ACTIONS:
        signature = effective_action_signature(action)
        aliases = by_signature.setdefault(signature, [])
        rows.append({
            "action_id": int(action["id"]),
            "action_name": action["name"],
            "effective_signature": signature,
            "aliased_with": list(aliases),
        })
        aliases.append(action["name"])
    exact_aliases = [names for names in by_signature.values() if len(names) > 1]
    if exact_aliases:
        raise ValueError(f"DQN actions have identical effective behavior: {exact_aliases}")
    return rows


def validate_action_space():
    names = [action["name"] for action in ACTIONS]
    ids = [int(action["id"]) for action in ACTIONS]
    if len(names) != len(set(names)):
        raise ValueError("DQN action names must be unique")
    if ids != list(range(len(ACTIONS))):
        raise ValueError("DQN action ids must be contiguous and match list order")
    valid_mutations = {"rand_1", "rand_2", "best_1", "current_to_best_1"}
    for action in ACTIONS:
        if not 0.0 <= float(action["F"]) <= 1.0:
            raise ValueError(f"Invalid F for DQN action {action['name']}")
        if not 0.0 <= float(action["CR"]) <= 1.0:
            raise ValueError(f"Invalid CR for DQN action {action['name']}")
        if tuple(sorted(action["repair_order"])) != tuple(sorted(REPAIR_STEPS)):
            raise ValueError(f"Invalid repair_order for DQN action {action['name']}")
        if action["mutation_strategy"] not in valid_mutations:
            raise ValueError(f"Invalid mutation strategy for DQN action {action['name']}")
    audit_action_space()
    return True


validate_action_space()