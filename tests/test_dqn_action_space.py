import pytest

from src.dqn.action_mask import MASK_RULE_ACTION_NAMES
from src.dqn.action_space import ACTIONS, get_action, validate_action_space


def test_dqn_action_space_contract_is_valid():
    assert validate_action_space()
    assert [action["id"] for action in ACTIONS] == list(range(len(ACTIONS)))
    assert len({action["name"] for action in ACTIONS}) == len(ACTIONS)
    assert MASK_RULE_ACTION_NAMES <= {action["name"] for action in ACTIONS}


def test_dqn_action_lookup_rejects_out_of_range_ids():
    with pytest.raises(IndexError):
        get_action(-1)
    with pytest.raises(IndexError):
        get_action(len(ACTIONS))
