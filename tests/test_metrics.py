import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.metrics import DecodeMetrics, StepRecord


def make_record(
    round_id,
    drafted,
    accepted,
    rejected=False,
    mode="speculative",
    draft_time=0.1,
    verify_time=0.2,
    commit_time=0.05,
    rollback_time=0.0,
    backpressure=False,
    stall=False,
):
    return StepRecord(
        round_id=round_id,
        drafted_tokens=drafted,
        accepted_tokens=accepted,
        rejected=rejected,
        corrective_token_added=rejected,
        committed_length_after_round=accepted,
        draft_time_sec=draft_time,
        verify_time_sec=verify_time,
        commit_time_sec=commit_time,
        rollback_penalty_sec=rollback_time,
        speculation_depth=drafted,
        round_acceptance=(accepted / drafted) if drafted else 0.0,
        mode_used=mode,
        speculative_tokens_peak=drafted,
        estimated_kv_overhead_units=10.0 * drafted,
        pipeline_busy_sec=draft_time + verify_time + commit_time,
        pipeline_bubble_sec=rollback_time,
        backpressure_event=backpressure,
        stall_event=stall,
    )


def test_decode_metrics_aggregate_acceptance_and_waste():
    metrics = DecodeMetrics(mode="fixed")
    metrics.generated_tokens = 5
    metrics.total_time_sec = 2.0

    metrics.add_step(make_record(1, drafted=4, accepted=3, rejected=True, rollback_time=0.01))
    metrics.add_step(make_record(2, drafted=2, accepted=2, rejected=False))

    assert metrics.rounds == 2
    assert metrics.draft_tokens_total == 6
    assert metrics.accepted_tokens_total == 5
    assert metrics.rollback_count == 1
    assert metrics.wasted_draft_tokens == 1
    assert metrics.acceptance_rate == 5 / 6
    assert metrics.tokens_per_second == 2.5
    assert metrics.peak_speculative_tokens == 4
    assert metrics.peak_estimated_kv_overhead_units == 40.0


def test_decode_metrics_baseline_fallback_is_counted_but_not_round_acceptance():
    metrics = DecodeMetrics(mode="hybrid_gated")
    metrics.add_step(make_record(1, drafted=4, accepted=4, mode="speculative"))
    metrics.add_step(make_record(2, drafted=0, accepted=0, mode="baseline_fallback"))

    assert metrics.baseline_fallback_steps == 1
    assert metrics.avg_round_acceptance == 1.0


def test_decode_metrics_bottleneck_and_pipeline_properties_are_stable():
    metrics = DecodeMetrics(mode="fixed")
    metrics.add_step(
        make_record(
            1,
            drafted=3,
            accepted=1,
            rejected=True,
            draft_time=0.1,
            verify_time=0.3,
            commit_time=0.1,
            rollback_time=0.5,
            backpressure=True,
            stall=True,
        )
    )

    assert metrics.backpressure_events == 1
    assert metrics.stall_rounds == 1
    assert 0.0 <= metrics.pipeline_utilization <= 1.0
    assert metrics.verify_bottleneck_ratio == 0.3 / 0.5
    assert metrics.rollback_share_ratio == 0.5 / 1.0
    assert metrics.energy_proxy_units > 0.0
    assert metrics.to_dict()["wasted_draft_tokens"] == 2
