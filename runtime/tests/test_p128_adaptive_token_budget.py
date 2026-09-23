from __future__ import annotations

import json
from copy import deepcopy

import pytest

from videoseek.core.token_budget import AdaptiveTokenBudget
from videoseek import utils
from videoseek import observer


def _append(path, total_tokens: int) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"total_tokens": total_tokens}) + "\n")


def test_soft_limit_allows_two_bounded_steps_then_stops(tmp_path):
    usage = tmp_path / "usage.jsonl"
    _append(usage, 99_000)
    budget = AdaptiveTokenBudget(
        {
            "adaptive_api_token_budget_enabled": True,
            "adaptive_api_token_soft_limit": 100_000,
            "adaptive_api_token_hard_limit": 150_000,
            "adaptive_api_token_soft_extra_steps": 2,
        },
        str(usage),
    )

    assert budget.observe(step_index=4)["soft_active"] is False
    _append(usage, 2_000)
    first = budget.observe(step_index=5)
    assert first["soft_active"] is True
    assert first["stop"] is False
    assert first["grace_steps_remaining"] == 2
    assert "do not repeat" in budget.planner_directive(step_index=5)

    second = budget.observe(step_index=6)
    assert second["stop"] is False
    assert second["grace_steps_remaining"] == 1

    stopped = budget.observe(step_index=7)
    assert stopped["stop"] is True
    assert stopped["stop_reason"] == "soft_limit_grace_exhausted"


def test_hard_limit_stops_without_waiting_for_soft_grace(tmp_path):
    usage = tmp_path / "usage.jsonl"
    _append(usage, 151_000)
    budget = AdaptiveTokenBudget(
        {
            "adaptive_api_token_budget_enabled": True,
            "adaptive_api_token_soft_limit": 100_000,
            "adaptive_api_token_hard_limit": 150_000,
            "adaptive_api_token_soft_extra_steps": 2,
        },
        str(usage),
    )

    status = budget.observe(step_index=3)
    assert status["stop"] is True
    assert status["stop_reason"] == "hard_investigation_limit"
    assert status["recorded_total_tokens"] == 151_000


def test_usage_reader_tolerates_partial_lines_and_disabled_budget(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(
        '{"total_tokens": 120000}\n{"total_tokens":', encoding="utf-8"
    )
    budget = AdaptiveTokenBudget(
        {"adaptive_api_token_budget_enabled": False}, str(usage)
    )

    status = budget.observe(step_index=19)
    assert status["recorded_total_tokens"] == 120_000
    assert status["stop"] is False


def test_zero_soft_extra_steps_is_honored(tmp_path):
    usage = tmp_path / "usage.jsonl"
    _append(usage, 100_000)
    budget = AdaptiveTokenBudget(
        {
            "adaptive_api_token_budget_enabled": True,
            "adaptive_api_token_soft_limit": 100_000,
            "adaptive_api_token_hard_limit": 150_000,
            "adaptive_api_token_soft_extra_steps": 0,
        },
        str(usage),
    )

    status = budget.observe(step_index=2)
    assert status["soft_extra_steps"] == 0
    assert status["stop"] is True
    assert status["stop_reason"] == "soft_limit_grace_exhausted"


def test_final_answer_is_single_claim_and_usage_is_separate(tmp_path):
    usage = tmp_path / "usage.jsonl"
    _append(usage, 151_000)
    budget = AdaptiveTokenBudget(
        {
            "adaptive_api_token_budget_enabled": True,
            "adaptive_api_token_soft_limit": 100_000,
            "adaptive_api_token_hard_limit": 150_000,
        },
        str(usage),
    )

    stopped = budget.observe(step_index=3)
    assert stopped["investigation_tokens_at_stop"] == 151_000
    assert budget.finish_reason() == (
        "adaptive_token_budget:hard_investigation_limit"
    )
    assert budget.begin_final_answer() is True
    assert budget.begin_final_answer() is False
    _append(usage, 777)
    completed = budget.complete_final_answer()
    assert completed["recorded_total_tokens"] == 151_777
    assert completed["investigation_tokens_at_stop"] == 151_000
    assert completed["final_answer_tokens"] == 777
    assert completed["final_answer_request_completed"] is True


def test_remote_request_guard_blocks_before_completion_dispatch(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise AssertionError("completion must not be reached")

    monkeypatch.setattr(utils, "completion", fake_completion)
    token = utils.install_api_request_budget_guard(lambda: False)
    try:
        with pytest.raises(utils.ApiRequestBudgetExceeded):
            utils.call_llm_api(
                model_name="test-model",
                messages=[{"role": "user", "content": "investigate"}],
                api_base="https://invalid.example/v1",
            )
    finally:
        utils.reset_api_request_budget_guard(token)

    assert calls == []


def test_observer_config_guard_covers_worker_thread_style_calls(monkeypatch):
    calls = []

    def fake_call_llm_api(**kwargs):
        calls.append(kwargs)
        raise AssertionError("observer must not dispatch")

    monkeypatch.setattr(observer, "call_llm_api", fake_call_llm_api)
    with pytest.raises(utils.ApiRequestBudgetExceeded):
        observer._call_api_observer(
            {
                "_adaptive_api_request_guard": lambda: False,
                "model_name": "test-model",
                "api_base": "https://invalid.example/v1",
                "api_key": "unused",
                "api_version": None,
                "max_tokens": 100,
                "reasoning_effort": "low",
                "seed": 42,
                "temperature": 1.0,
            },
            content=[{"type": "text", "text": "inspect"}],
        )

    assert calls == []


def test_request_guard_survives_tool_config_deepcopy_without_copying_owner():
    owner = object()
    guard = utils.ApiRequestBudgetGuard(lambda: owner is not None)
    copied = deepcopy({"_adaptive_api_request_guard": guard})

    assert copied["_adaptive_api_request_guard"] is guard
    assert copied["_adaptive_api_request_guard"]() is True
