import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cache import KVCacheManager


def test_commit_accepted_prefix_rejects_negative_count():
    cache = KVCacheManager(committed_length=5, speculative_length=2)

    with pytest.raises(ValueError):
        cache.commit_accepted_prefix(-1)


def test_commit_accepted_prefix_rejects_count_larger_than_speculative_length():
    cache = KVCacheManager(committed_length=5, speculative_length=2)

    with pytest.raises(ValueError):
        cache.commit_accepted_prefix(3)


def test_reset_to_checkpoint_restores_committed_and_speculative_lengths():
    cache = KVCacheManager(committed_length=10)
    checkpoint = cache.checkpoint()

    cache.append_speculative_token()
    cache.append_speculative_token()
    cache.commit_accepted_prefix(1)
    cache.commit_corrective_token()

    assert cache.committed_length != checkpoint.committed_length

    cache.reset_to_checkpoint(checkpoint)

    assert cache.committed_length == 10
    assert cache.speculative_length == 0
    assert cache.history[-1].event == "reset_to_checkpoint"
