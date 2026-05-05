import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import DecodeConfig, ExperimentConfig


def test_decode_config_defaults_are_safe_for_quick_runs():
    cfg = DecodeConfig()

    assert cfg.max_new_tokens == 50
    assert cfg.speculation_depth == 4
    assert cfg.do_sample is False
    assert cfg.temperature == 1.0


def test_experiment_config_defaults_use_small_gpt2_family():
    cfg = ExperimentConfig()

    assert cfg.draft_model_name == "distilgpt2"
    assert cfg.target_model_name == "gpt2"
    assert cfg.device in {"cuda", "cpu", "mps"}
