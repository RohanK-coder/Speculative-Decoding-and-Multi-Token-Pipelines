import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.speculative import speculative_greedy_decode


class ScriptedModel:
    """Tiny deterministic model used to test decoding logic without downloads."""

    def __init__(self, next_tokens, verify_outputs=None):
        self.next_tokens = list(next_tokens)
        self.verify_outputs = verify_outputs or []
        self.greedy_calls = 0
        self.verify_calls = 0

    def encode(self, prompt):
        return torch.tensor([[0]], dtype=torch.long)

    def decode(self, input_ids):
        return " ".join(str(x) for x in input_ids[0].tolist())

    def greedy_next_token(self, input_ids):
        idx = min(self.greedy_calls, len(self.next_tokens) - 1)
        self.greedy_calls += 1
        return self.next_tokens[idx]

    def verify_block(self, prefix_ids, draft_ids):
        if self.verify_calls < len(self.verify_outputs):
            output = self.verify_outputs[self.verify_calls]
        else:
            output = draft_ids
        self.verify_calls += 1
        return output[: len(draft_ids)]


def test_speculative_commits_all_tokens_when_target_agrees():
    draft = ScriptedModel(next_tokens=[1, 2, 3, 4])
    target = ScriptedModel(next_tokens=[99], verify_outputs=[[1, 2, 3, 4]])

    _, ids, metrics, cache = speculative_greedy_decode(
        draft_model=draft,
        target_model=target,
        prompt="ignored",
        max_new_tokens=4,
        speculation_depth=4,
    )

    assert ids.tolist() == [[0, 1, 2, 3, 4]]
    assert metrics.accepted_tokens_total == 4
    assert metrics.rollback_count == 0
    assert metrics.acceptance_rate == 1.0
    assert cache.speculative_length == 0


def test_speculative_rolls_back_rejected_suffix_and_adds_corrective_token():
    draft = ScriptedModel(next_tokens=[1, 2, 3, 4])
    target = ScriptedModel(next_tokens=[8], verify_outputs=[[1, 8, 9, 9]])

    _, ids, metrics, cache = speculative_greedy_decode(
        draft_model=draft,
        target_model=target,
        prompt="ignored",
        max_new_tokens=2,
        speculation_depth=4,
    )

    assert ids.tolist() == [[0, 1, 8]]
    assert metrics.accepted_tokens_total == 1
    assert metrics.rollback_count == 1
    assert metrics.wasted_draft_tokens == 3
    assert metrics.step_records[0].corrective_token_added is True
    assert cache.speculative_length == 0


def test_hybrid_gated_uses_baseline_fallback_after_low_acceptance():
    draft = ScriptedModel(next_tokens=[1, 2, 3, 4])
    target = ScriptedModel(
        next_tokens=[7, 8],
        verify_outputs=[[6, 6]],
    )

    _, ids, metrics, _ = speculative_greedy_decode(
        draft_model=draft,
        target_model=target,
        prompt="ignored",
        max_new_tokens=2,
        speculation_depth=2,
        hybrid_gated=True,
        gate_threshold=0.9,
    )

    assert ids.shape[1] == 3
    assert metrics.rollback_count == 1
    assert metrics.baseline_fallback_steps >= 1
    assert any(r.mode_used == "baseline_fallback" for r in metrics.step_records)
