from transport import ReplayContractError
from typing import Mapping, Any
class P133UnresolvedDecision(ReplayContractError): pass
def _finish_reason_matches(value, base): return value == base or value.startswith(base + ":")
def validate_terminal_state(
    global_state: Mapping[str, Any], row: Mapping[str, Any]
) -> str:
    """Return validated/forced only for a fully consistent terminal record."""

    terminal = str(global_state.get("terminal_status") or "")
    decision = global_state.get("answer_decision") or {}
    snapshot = global_state.get("snapshot") or {}
    prediction = str(row.get("pred_letter") or "").strip().upper()
    raw_prediction = str(row.get("pred_raw") or "").strip().upper()
    finish_reason = str(row.get("finish_reason") or "")

    if terminal == "validated_answer":
        answer = str(global_state.get("validated_answer") or "").strip().upper()
        valid = (
            answer in {"A", "B", "C", "D"}
            and global_state.get("evaluation_prediction") == answer
            and global_state.get("forced_prediction") is False
            and decision.get("status") == "validated"
            and decision.get("selected_option") == answer
            and decision.get("validated") is True
            and decision.get("forced") is False
            and snapshot.get("decision_sufficient") is True
            and snapshot.get("validated_option") == answer
            and prediction == answer
            and raw_prediction == answer
            and _finish_reason_matches(finish_reason, "p133_validated")
        )
        if not valid:
            raise ReplayContractError("inconsistent P132 validated terminal state")
        return "validated"

    if terminal == "forced_unresolved":
        raise P133UnresolvedDecision("P132 ended without a forced evaluation prediction")

    if terminal == "forced_prediction":
        answer = str(global_state.get("evaluation_prediction") or "").strip().upper()
        if (
            not answer
            or decision.get("status") == "forced_unresolved"
            or decision.get("selected_option") is None
        ):
            raise P133UnresolvedDecision("P132 ended without a forced evaluation prediction")
        valid = (
            answer in {"A", "B", "C", "D"}
            and global_state.get("validated_answer") is None
            and global_state.get("forced_prediction") is True
            and decision.get("status") == "forced"
            and decision.get("selected_option") == answer
            and decision.get("validated") is False
            and decision.get("forced") is True
            and snapshot.get("decision_sufficient") is False
            and snapshot.get("validated_option") is None
            and prediction == answer
            and raw_prediction == answer
            and _finish_reason_matches(finish_reason, "p133_forced_prediction")
        )
        if not valid:
            raise ReplayContractError("inconsistent P132 forced terminal state")
        return "forced"

    raise ReplayContractError(f"unknown P132 terminal status: {terminal!r}")

