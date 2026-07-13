"""Discrete action space for DQN-CR-MODE."""

ACTIONS = [
    {"id": 0, "name": "balanced", "F": 0.5, "CR": 0.8, "mutation_strategy": "rand_1", "repair_order": ["deploy", "link", "capacity", "energy", "service", "sink"], "power_policy": "balanced"},
    {"id": 1, "name": "exploration_high_F", "F": 0.8, "CR": 0.9, "mutation_strategy": "rand_1", "repair_order": ["deploy", "link", "capacity", "energy", "service", "sink"], "power_policy": "balanced"},
    {"id": 2, "name": "exploitation_low_F", "F": 0.35, "CR": 0.6, "mutation_strategy": "current_to_best_1", "repair_order": ["deploy", "link", "capacity", "energy", "service", "sink"], "power_policy": "balanced"},
    {"id": 3, "name": "energy_first", "F": 0.45, "CR": 0.7, "mutation_strategy": "rand_1", "repair_order": ["energy", "sink", "link", "capacity", "service", "deploy"], "power_policy": "conservative"},
    {"id": 4, "name": "link_first", "F": 0.55, "CR": 0.8, "mutation_strategy": "rand_1", "repair_order": ["link", "capacity", "energy", "sink", "service", "deploy"], "power_policy": "balanced"},
    {"id": 5, "name": "ap_load_first", "F": 0.5, "CR": 0.75, "mutation_strategy": "rand_1", "repair_order": ["capacity", "link", "energy", "sink", "service", "deploy"], "power_policy": "balanced"},
    {"id": 6, "name": "sink_first", "F": 0.45, "CR": 0.65, "mutation_strategy": "rand_1", "repair_order": ["sink", "energy", "link", "capacity", "service", "deploy"], "power_policy": "sink_limited"},
    {"id": 7, "name": "rsum_capacity_priority", "F": 0.6, "CR": 0.85, "mutation_strategy": "best_1", "repair_order": ["link", "capacity", "energy", "sink", "service", "deploy"], "power_policy": "rsum_capacity_priority"},
    {"id": 8, "name": "diversity_boost", "F": 0.9, "CR": 0.95, "mutation_strategy": "rand_2", "repair_order": ["deploy", "link", "capacity", "energy", "service", "sink"], "power_policy": "balanced"},
    {"id": 9, "name": "conservative_repair", "F": 0.3, "CR": 0.5, "mutation_strategy": "current_to_best_1", "repair_order": ["energy", "capacity", "link", "sink", "service", "deploy"], "power_policy": "conservative"},
]


def get_action(action_id):
    return ACTIONS[int(action_id) % len(ACTIONS)]


def action_dim():
    return len(ACTIONS)
