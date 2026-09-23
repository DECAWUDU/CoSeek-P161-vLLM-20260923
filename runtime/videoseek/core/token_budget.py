from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AdaptiveTokenBudget:
    """Track recorded API usage and stop open-ended investigation loops.

    The limits apply to investigation.  A single compact final-answer request is
    still allowed after a stop so a completed evaluation row is produced.
    """

    def __init__(self, config: dict[str, Any], usage_log: str | None):
        self.enabled = bool(config.get("adaptive_api_token_budget_enabled", False))

        def configured_int(name: str, default: int) -> int:
            # ``0`` is meaningful for soft_extra_steps.  Using ``value or
            # default`` silently changed an explicitly configured zero to two.
            value = config.get(name)
            return int(default if value is None else value)

        self.soft_limit = max(
            1, configured_int("adaptive_api_token_soft_limit", 100_000)
        )
        self.hard_limit = max(
            self.soft_limit,
            configured_int("adaptive_api_token_hard_limit", 150_000),
        )
        self.soft_extra_steps = max(
            0, configured_int("adaptive_api_token_soft_extra_steps", 2)
        )
        self.final_answer_max_tokens = max(
            64,
            configured_int("adaptive_api_token_final_answer_max_tokens", 4096),
        )
        # Cover a full final Capsule in addition to the observed prompt overhead.
        # This intentionally over-reserves when that prompt already contained a Capsule.
        self.final_capsule_tokens = max(0, int(config.get("coseek1_planner_capsule_token_budget") or 4000)) if (
            config.get("coseek1_planner_decision_neutral_capsule_enabled")
            or config.get("coseek1_planner_evidence_capsule_enabled")) else 0
        self.usage_log = Path(usage_log) if usage_log else None
        self.soft_reached_step: int | None = None
        self.stop_step: int | None = None
        self.stop_reason: str | None = None
        self.last_total_tokens = 0
        self.largest_request_tokens = 0
        self.largest_text_prompt_tokens = 0
        self.investigation_tokens_at_stop: int | None = None
        self.final_answer_request_started = False
        self.final_answer_request_completed = False
        self.final_answer_tokens = 0
        self._final_answer_start_tokens: int | None = None
        self.events: list[dict[str, Any]] = []

    def read_total_tokens(self) -> int:
        if self.usage_log is None or not self.usage_log.is_file():
            self.last_total_tokens = 0
            return 0
        total = 0
        for line in self.usage_log.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict):
                continue
            value = row.get("total_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                total += value
                # Use observed request costs, not a per-image pricing formula.
                if isinstance(row.get("prompt_tokens"), int):
                    self.largest_request_tokens = max(self.largest_request_tokens, value)
                    if row.get("has_images") is False:
                        self.largest_text_prompt_tokens = max(self.largest_text_prompt_tokens, row["prompt_tokens"])
        self.last_total_tokens = total
        return total

    def observe(self, *, step_index: int) -> dict[str, Any]:
        total = self.read_total_tokens()
        if not self.enabled:
            return self.snapshot(step_index=step_index)

        if total >= self.soft_limit and self.soft_reached_step is None:
            self.soft_reached_step = int(step_index)
            self.events.append(
                {
                    "event": "soft_limit_reached",
                    "step": int(step_index) + 1,
                    "total_tokens": total,
                }
            )

        reason = None
        if total >= self.hard_limit:
            reason = "hard_investigation_limit"
        elif (self.largest_request_tokens
              and total + self.largest_request_tokens + self.final_answer_reserve_tokens > self.hard_limit):
            reason = "final_answer_reserve"
        elif (
            self.soft_reached_step is not None
            and int(step_index) - self.soft_reached_step >= self.soft_extra_steps
        ):
            reason = "soft_limit_grace_exhausted"

        if reason is not None and self.stop_reason is None:
            self.stop_reason = reason
            self.stop_step = int(step_index)
            self.investigation_tokens_at_stop = total
            self.events.append(
                {
                    "event": "investigation_stopped",
                    "reason": reason,
                    "step": int(step_index) + 1,
                    "total_tokens": total,
                }
            )
        return self.snapshot(step_index=step_index)

    @property
    def final_answer_reserve_tokens(self) -> int:
        # Advisory headroom: an unseen larger request can still exceed estimates.
        return self.largest_text_prompt_tokens + self.final_capsule_tokens + self.final_answer_max_tokens

    def begin_final_answer(self) -> bool:
        """Claim the sole compact final-answer request after a budget stop."""

        if not self.enabled or self.stop_reason is None:
            return True
        if self.final_answer_request_started:
            return False
        self.read_total_tokens()
        if self.investigation_tokens_at_stop is None:
            self.investigation_tokens_at_stop = self.last_total_tokens
        self.final_answer_request_started = True
        self._final_answer_start_tokens = self.last_total_tokens
        self.events.append(
            {
                "event": "compact_final_answer_started",
                "total_tokens": self.last_total_tokens,
            }
        )
        return True

    def complete_final_answer(self) -> dict[str, Any]:
        """Record usage added by the claimed compact final-answer request."""

        self.read_total_tokens()
        if self.final_answer_request_started:
            start = int(self._final_answer_start_tokens or 0)
            self.final_answer_tokens = max(0, self.last_total_tokens - start)
            if not self.final_answer_request_completed:
                self.events.append(
                    {
                        "event": "compact_final_answer_completed",
                        "total_tokens": self.last_total_tokens,
                        "final_answer_tokens": self.final_answer_tokens,
                    }
                )
            self.final_answer_request_completed = True
        return self.snapshot()

    def finish_reason(self) -> str | None:
        if self.stop_reason is None:
            return None
        return f"adaptive_token_budget:{self.stop_reason}"

    def snapshot(self, *, step_index: int | None = None) -> dict[str, Any]:
        soft_active = self.enabled and self.soft_reached_step is not None
        grace_remaining = None
        if soft_active and step_index is not None:
            grace_remaining = max(
                0,
                self.soft_extra_steps - (int(step_index) - self.soft_reached_step),
            )
        return {
            "enabled": self.enabled,
            "soft_limit_tokens": self.soft_limit,
            "hard_investigation_limit_tokens": self.hard_limit,
            "soft_extra_steps": self.soft_extra_steps,
            "final_answer_max_tokens": self.final_answer_max_tokens,
            "recorded_total_tokens": self.last_total_tokens,
            "next_request_estimate_tokens": self.largest_request_tokens,
            "final_answer_reserve_tokens": self.final_answer_reserve_tokens,
            "investigation_tokens_at_stop": self.investigation_tokens_at_stop,
            "final_answer_request_started": self.final_answer_request_started,
            "final_answer_request_completed": self.final_answer_request_completed,
            "final_answer_tokens": self.final_answer_tokens,
            "soft_active": soft_active,
            "soft_reached_step": (
                self.soft_reached_step + 1
                if self.soft_reached_step is not None
                else None
            ),
            "grace_steps_remaining": grace_remaining,
            "stop": self.stop_reason is not None,
            "stop_reason": self.stop_reason,
            "stop_step": self.stop_step + 1 if self.stop_step is not None else None,
            "events": list(self.events),
        }

    def planner_directive(self, *, step_index: int) -> str:
        status = self.snapshot(step_index=step_index)
        if not status["soft_active"] or status["stop"]:
            return ""
        return (
            "Adaptive token budget: the soft investigation limit has been reached "
            f"({status['recorded_total_tokens']} recorded tokens; "
            f"{status['grace_steps_remaining']} bounded step(s) remain). Prefer an "
            "answer from existing verified evidence. Use at most one narrowly targeted, "
            "non-overlapping check only when it can resolve the leading hypothesis; do "
            "not repeat a covered window or launch another broad search.\n\n"
        )
