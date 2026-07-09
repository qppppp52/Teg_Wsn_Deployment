import copy

import pytest

from src.rl_init.checkpoint import check_checkpoint_compatibility, save_checkpoint, load_checkpoint
from tests.test_rl_init_checkpoint import TinyAgent


torch = pytest.importorskip("torch")


BASE_META = {
    "scene_name": "small_center_heat_compare",
    "seed": 42,
    "throughput_metric": "actual",
    "num_candidates": 5,
    "candidate_feature_dim": 3,
    "global_feature_dim": 4,
    "hidden_dim": 16,
    "action_roles": ["sensor", "ap", "skip", "stop"],
    "max_selected_sensors": 3,
    "max_selected_aps": 1,
    "config_hash": "abc123",
}


def test_checkpoint_compatible_when_metadata_matches():
    compatible, reason = check_checkpoint_compatibility(BASE_META, BASE_META)
    assert compatible is True
    assert reason == ""


@pytest.mark.parametrize("key,value", [
    ("candidate_feature_dim", 5),
    ("throughput_metric", "capacity"),
    ("scene_name", "other_scene"),
])
def test_checkpoint_incompatible_when_metadata_differs(key, value):
    expected = copy.deepcopy(BASE_META)
    expected[key] = value
    compatible, reason = check_checkpoint_compatibility(BASE_META, expected)
    assert compatible is False
    assert reason == f"{key}_mismatch"


def test_incompatible_checkpoint_is_not_loaded():
    from pathlib import Path
    path = Path("test_outputs") / "checkpoint_incompatible" / "policy.pt"
    save_checkpoint(TinyAgent(), str(path), meta=BASE_META)
    expected = copy.deepcopy(BASE_META)
    expected["throughput_metric"] = "capacity"
    result = load_checkpoint(TinyAgent(), str(path), expected_meta=expected)
    assert result["loaded"] is False
    assert result["compatible"] is False
    assert result["skip_reason"] == "throughput_metric_mismatch"
