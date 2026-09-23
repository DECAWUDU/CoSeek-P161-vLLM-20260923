import json
import os
import re
import time
from copy import deepcopy
from typing import Any, List
from abc import ABC, abstractmethod

from videoseek.video_reader import VideoReader

from .core import (
    Action,
    Observation,
    Trajectory,
    TrajectoryStep,
    format_evidence_for_answer,
    format_memory_for_prompt,
    init_observation_memory,
    merge_tool_observation,
    matching_episode_for_range,
    record_answer_evidence_audit,
    record_episode_planner_proposal,
    record_planner_proposal,
    record_routing_audit,
    refresh_candidate_frontier,
    refresh_evidence_episode_frontier,
)
from .core.memory import build_event_coverage, extract_v10_payload
from .core.minimal_global_fsm import (
    detect_global_mode,
    force_answer as p130_force_answer,
    should_use_minimal_global_fsm,
    validate_answer as p130_validate_answer,
)
from .core.p130_runtime import (
    append_p130_observation,
    format_p130_snapshot,
    init_p130_state,
    GLOBAL_PLANNER_INSTRUCTION,
    global_planner_state,
    global_planner_action,
    store_global_decision,
    refresh_p130_snapshot,
)
from .core.observation_reuse import (
    find_reusable_observation,
    make_cache_entry,
    reused_payload,
)
from .core.overview_admissibility import (
    canonical_action_signature,
    format_blocked_overview_feedback,
    has_valid_overview_receipt,
    record_blocked_overview,
    record_overview_attempt,
)
from .core.planner_capsule import (
    build_decision_neutral_planner_capsule,
    build_planner_evidence_capsule,
    estimate_tokens,
)
from .core.token_budget import AdaptiveTokenBudget
from .tools import DEFAULT_TOOL_REGISTRY
from .tools.v10_format import format_v10_observation
from .utils import (
    ApiRequestBudgetGuard,
    ApiRequestBudgetExceeded,
    call_llm_api,
    convert_to_free_form_text_representation,
    extract_json_object,
    install_api_request_budget_guard,
    load_subtitles,
    SUBTITLE_WINDOW_CONTRACT,
    planner_subtitle_view,
    reset_api_request_budget_guard,
)
from .skeleton import get_scene_context_for_window


class BaseAgent(ABC):
    def __init__(self) -> None:
        self.messages: List[dict] = []
        self.final_answer = None
        self.question = None

    def reset(self) -> None:
        self.messages = self.construct_initial_messages()
        self.final_answer = None
        self.question = None

    @abstractmethod
    def construct_initial_messages(self) -> List[dict]:
        raise NotImplementedError

    @abstractmethod
    def run(self, question: str) -> Trajectory:
        raise NotImplementedError


class VideoSeekAgent(BaseAgent):
    def __init__(
        self,
        config: dict,
        video_path: str,
        subtitle_path: str,
        output_dir: str,
        tools: list,
        verbose: bool = False,
    ):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.vr = VideoReader(video_path)
        self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.tools = self.tool_registry.resolve_tools(tools + ["answer"])
        self.allowed_tool_names = {
            tool["function"]["name"]
            for tool in self.tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        self.output_dir = output_dir
        self.verbose = verbose

        self.duration = round(len(self.vr) / self.vr.get_avg_fps(), 2)
        self.subtitles = load_subtitles(subtitle_path)

        # LLM config
        self.model_name = config["model_name"]
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.api_version = config["api_version"]
        self.max_steps = config["max_steps"]
        self.max_tokens = config["max_tokens"]
        self.reasoning_effort = config["reasoning_effort"]
        self.seed = config["seed"]
        self.temperature = config["temperature"]

        # initial messages
        self.messages = self.construct_initial_messages()
        # trajectory
        self.trajectory_steps: List[TrajectoryStep] = []
        self.observation_memory = init_observation_memory()
        self.__record_v34_runtime_config()
        self._qwen_first_api_fallbacks: dict[str, Action] = {}
        self._query_scene_probe_count = 0
        self._query_scene_probe_scene_ids: set[str] = set()
        self._candidate_router_reason = ""
        self._verified_observation_cache: list[dict[str, Any]] = []
        self._adaptive_token_budget = AdaptiveTokenBudget(
            self.config, os.environ.get("COSEEK_USAGE_LOG")
        )
        self._adaptive_budget_status = self._adaptive_token_budget.snapshot()
        self._adaptive_budget_step = 0
        self._adaptive_final_answer_request_active = False

    def reset(self):
        """
        Reset the agent.
        """
        super().reset()
        self.trajectory_steps = []
        self.observation_memory = init_observation_memory()
        self.__record_v34_runtime_config()
        self._qwen_first_api_fallbacks = {}
        self._query_scene_probe_count = 0
        self._query_scene_probe_scene_ids = set()
        self._candidate_router_reason = ""
        self._verified_observation_cache = []
        self._adaptive_token_budget = AdaptiveTokenBudget(
            self.config, os.environ.get("COSEEK_USAGE_LOG")
        )
        self._adaptive_budget_status = self._adaptive_token_budget.snapshot()
        self._adaptive_budget_step = 0
        self._adaptive_final_answer_request_active = False

    def __record_v34_runtime_config(self) -> None:
        keys = (
            "coseek1_scope_aware_evidence_memory",
            "coseek1_separate_routing_candidates",
            "coseek1_persistent_open_gaps",
            "coseek1_semantic_gap_resolution_enabled",
            "structured_evidence_preserve_aggregate_verifier_decision",
            "coseek1_observer_conflict_memory_enabled",
            "coseek1_query_relevant_memory_retention",
            "coseek1_persistent_candidate_pool",
            "coseek1_persistent_candidate_pool_max_items",
            "coseek1_persistent_candidate_pool_timestamp_radius_s",
            "coseek1_candidate_evidence_projection",
            "coseek1_planner_evidence_capsule_enabled",
            "coseek1_planner_decision_neutral_capsule_enabled",
            "coseek1_planner_capsule_token_budget",
            "coseek1_planner_capsule_max_verified",
            "coseek1_planner_capsule_max_candidates",
            "coseek1_planner_capsule_max_obligations",
            "coseek1_planner_capsule_audit_enabled",
            "coseek1_valid_overview_receipt_guard_enabled",
            "coseek1_candidate_context_localize_enabled",
            "focus_qwen_multi_window_enabled",
            "focus_qwen_multi_window_max_windows",
            "overview_require_all_timestamps",
            "overview_fill_omitted_timestamps",
            "overview_frame_caption_words",
            "overview_frame_short_side",
            "overview_embed_timestamp_labels",
            "overview_contact_sheet_group_size",
            "dual_path_overview_enabled",
            "dual_path_overview_shadow_only",
            "dual_path_integration_mode",
            "dual_path_local_frames",
            "dual_path_local_min_frames",
            "dual_path_local_max_frames",
            "dual_path_local_target_interval_s",
            "dual_path_local_batch_size",
            "dual_path_local_min_processed_frames",
            "dual_path_local_top_k",
            "dual_path_local_candidate_pool_k",
            "dual_path_fused_top_k",
            "dual_path_semantic_prototype_ranking_enabled",
            "dual_path_planner_caption_limit",
            "dual_path_local_deadline_s",
            "dual_path_local_caption_words",
            "coseek1_direct_planner_answer",
            "coseek1_localize_qwen_enabled",
            "localize_qwen_top_k",
            "localize_qwen_coarse_max_frames",
            "localize_qwen_fine_max_frames",
            "localize_qwen_verify_window_s",
            "localize_qwen_verify_context_margin_s",
            "localize_qwen_search_context_margin_s",
            "localize_qwen_adaptive_search_context",
            "localize_qwen_normalized_candidate_score_enabled",
            "localize_qwen_evidence_preserving_selection_enabled",
            "localize_qwen_ambiguous_reserve",
            "localize_qwen_duration_budget_enabled",
            "localize_qwen_duration_budget_fps",
            "localize_qwen_codec_anchor_sampling_enabled",
            "localize_qwen_boundary_expansion_enabled",
            "localize_qwen_caption_cache_enabled",
            "localize_qwen_candidate_novelty_enabled",
            "coseek1_localize_inline_verify",
            "localize_inline_verify_max_windows",
            "localize_inline_verify_max_frames",
            "localize_inline_verify_batch_enabled",
            "localize_inline_verify_batch_size",
            "localize_inline_verify_contact_sheets",
            "multiwindow_verify_recovery_enabled",
            "multiwindow_verify_recovery_max_calls",
            "multiwindow_verify_recovery_batch_size",
            "grounded_direct_verify_packet_enabled",
            "grounded_direct_verify_packet_group_size",
            "grounded_direct_verify_packet_anchor_count",
            "grounded_direct_verify_packet_detail_max_side",
            "grounded_verify_packet_temporal_anchor_spread_enabled",
            "packet_adaptive_evidence_need_enabled",
            "packet_multiwindow_decision_sufficiency_enabled",
            "packet_detail_grounding_direct_escalation_enabled",
            "coseek1_compact_event_coverage",
            "coseek1_compact_event_coverage_max_items",
            "coseek1_event_coverage_inline_verify",
            "coseek1_event_coverage_inline_verify_max_windows",
            "coseek1_event_coverage_inline_verify_radius_s",
            "coseek1_temporal_evidence_ledger_enabled",
            "coseek1_temporal_evidence_ledger_max_events",
            "p125_per_candidate_direct_evidence_enabled",
            "p125_ledger_capsule_bridge_enabled",
            "p127_strict_per_anchor_evidence_enabled",
            "p127_ambiguous_mandatory_detail_recovery_enabled",
            "adaptive_api_token_budget_enabled",
            "adaptive_api_token_soft_limit",
            "adaptive_api_token_hard_limit",
            "adaptive_api_token_soft_extra_steps",
            "adaptive_api_token_final_answer_max_tokens",
            "p130_minimal_global_fsm_enabled",
            "p131_minimal_global_repairs_enabled",
            "p132_weak_lead_visible_core_enabled",
            "p130_global_recovery_max_attempts",
            "p130_global_initial_max_search_windows",
            "coseek1_verified_observation_reuse_enabled",
            "coseek1_verified_observation_reuse_overlap",
            "coseek1_verified_observation_reuse_query_similarity",
            "coseek1_tool_integrity_repair_enabled",
            "coseek1_tool_memory_anchor_margin_s",
            "coseek1_tool_memory_anchors_per_window",
            "coseek1_tool_memory_anchor_pad_s",
            "coseek1_tool_event_identity_repair_enabled",
            "coseek1_tool_recollect_expanded_anchors_enabled",
            "grounded_verify_packet_boundary_context_enabled",
        )
        self.observation_memory["runtime_config"] = {
            key: self.config.get(key) for key in keys
        }

    def construct_initial_messages(self) -> List[dict]:
        """
        Build the initial [system, user] messages.
        Subclasses should return a list of chat messages that may contain placeholders.
        """
        system_prompt = self.config["SYSTEM_PROMPT"].format(
            overview_num_frames=self.config["frame_sampling_factor"] * self.config["overview_base"],
            skim_num_frames=self.config["frame_sampling_factor"] * self.config["skim_base"],
            focus_num_frames=self.config["frame_sampling_factor"] * self.config["focus_base"],
        )
        qwen_first = bool(self.config.get("qwen_first_observer", False))
        if qwen_first and not self.config.get("coseek1_structured_planner", False):
            system_prompt += (
                "\n\n  ## CoSeek Qwen-first visual observation policy\n"
                "  - GPT plans and reasons over trajectory; local Qwen is used as the cheap first visual observer for eligible skim/focus windows.\n"
                "  - The router may execute skim_qwen/focus_qwen when you request skim/focus, and may upgrade uncertain Qwen observations to API verification.\n"
                "  - Treat Qwen observations as timestamped routing evidence. If Qwen reports missing detail or weak evidence, ask for a stronger focus/skim instead of answering prematurely.\n"
            )
        if (
            not self.config.get("coseek1_structured_planner", False)
            and (qwen_first or "focus_qwen" in (self.config.get("tools") or []))
        ):
            system_prompt += (
                "\n\n  ## `focus_qwen`: cheap local Qwen inspection of a short clip\n"
                "  - When: Use this instead of `focus` only for simple visual checks in a short, evidence-based window: object/person/color/scene presence, rough visible count, or option verification.\n"
                "  - How: It samples a few real frames, packs them into compact 2x2 sheets, and returns a very short structured observation.\n"
                "  - Constraints: Do not use `focus_qwen` for timeline, occurrence counting, before/after motion, collision/fall/action order, or final reasoning. Use `focus` for those.\n"
                "  - Treat `focus_qwen` as weak local evidence; if it is unclear or conflicts with prior memory, follow up with `focus`.\n"
                "\n  Tool Calling Policy addition:\n"
                "  - Prefer `focus_qwen` for cheap verification of static/fine visual details in one short window.\n"
                "  - Prefer `focus` for high-stakes evidence, temporal motion, count/order reasoning, and when exact visual proof is required.\n"
            )
        if (
            not self.config.get("coseek1_structured_planner", False)
            and (qwen_first or "skim_qwen" in (self.config.get("tools") or []))
        ):
            system_prompt += (
                "\n\n  ## `skim_qwen`: cheap local Qwen coarse scan of a candidate window\n"
                "  - When: Use this before `focus` for low-cost rough localization inside a candidate scene/window, especially for normal, option_coverage, or alternative_window checks.\n"
                "  - How: It samples a few frames across the window, packs them into compact 2x2 sheets, and returns rough_event, scene_tags, relevance, and a suggested focus window.\n"
                "  - Constraints: Do not use `skim_qwen` for occurrence counting, full-video anomaly judgment, precise event order, or final reasoning. Use `skim` for those.\n"
                "  - Treat `skim_qwen` as weak routing memory; verify important evidence with `focus_qwen` or `focus`.\n"
                "\n  Tool Calling Policy addition:\n"
                "  - Prefer `skim_qwen` to cheaply inspect a medium candidate window before spending an API `skim`/`focus` call.\n"
                "  - Prefer `skim` when the question needs timeline, count_occurrence, broad coverage, or high-recall observation.\n"
            )
        if self.config.get("coseek1_frontier_investigation_state", False):
            system_prompt += (
                "\n\n  ## Investigation Frontier\n"
                "  - For each next observation, bind one Candidate Frontier id to the unresolved "
                "difference between the leading hypothesis and strongest competitor.\n"
                "  - Prefer a candidate whose `tests` field can change that comparison, rather "
                "than a nearby window that only repeats already observed behavior.\n"
                "  - This is investigation state, not a forced route: you may choose another "
                "candidate, but explain the expected new information.\n"
                "  - When answering, use candidate_id=null and state why no remaining listed "
                "discriminator is likely to change the result.\n"
            )
        if self.config.get("coseek1_evidence_episode_frontier", False):
            system_prompt += (
                "\n\n  ## Evidence Episode Frontier\n"
                "  - Treat each episode as a local event hypothesis with timestamps, provenance, "
                "inspection status, and unresolved boundaries.\n"
                "  - Use one logical skim_qwen for a broad unresolved episode, focus_qwen for a "
                "precise timestamp anchor, and frame_verify only for the final object, relation, "
                "or option distinction.\n"
                "  - A verified short episode is local evidence; it cannot eliminate a competing "
                "episode elsewhere in the video.\n"
                "  - Bind the next observation to one episode_id, but retain autonomy to choose a "
                "different episode when it offers more discriminating information.\n"
                "  - Answer Evidence Audit is logging only and never prevents answer().\n"
            )
        if self.config.get("coseek1_temporal_evidence_ledger_enabled", False):
            system_prompt += (
                "\n\n  ## Temporal Evidence Ledger\n"
                "  - Treat each TE row as a stable, deduplicated span of already verified visual evidence.\n"
                "  - An event may unfold across adjacent shots; require simultaneous same-frame visibility only when the question explicitly asks for simultaneity or one shot.\n"
                "  - The number of TE rows is observed event coverage, not automatically the final answer because unseen events may remain.\n"
                "  - Contextual candidates are neither verified occurrences nor hard negatives; use the plausible observed range when the task's definition may count a surrounding activity segment rather than only the sampled peak action.\n"
                "  - decision_signal=option_discriminative means that verified evidence already separates the options; inspect again only when a distinct candidate or a verified conflict could change that conclusion.\n"
                "  - If a tool returns no_new_visual_information=true, rephrasing the same request cannot reveal new pixels; use the ledger, inspect a temporally distinct candidate, or answer from current evidence.\n"
            )
        return [{"role": "system", "content": system_prompt}]

    def __parse_actions(self, thought: str) -> List[Action]:
        """
        Parse the actions from the thought.
        """
        if self.config.get("coseek1_planner_json_action", False):
            json_actions = self.__extract_planner_json_actions(thought)
            if json_actions:
                return self.__normalize_coseek1_actions(json_actions)

        explicit_actions = self.__extract_explicit_thought_actions(thought)
        if self.config.get("coseek1_structured_planner", False):
            explicit_actions = self.__normalize_coseek1_actions(explicit_actions)
            if explicit_actions:
                return explicit_actions
        explicit_has_observation = any(
            action.function_name != "answer" for action in explicit_actions
        )
        if self.config.get("direct_answer_from_letter_thought", True) and not explicit_has_observation:
            final_letter = self.__extract_final_answer_letter(thought)
            if final_letter:
                return [
                    Action(
                        function_name="answer",
                        parameters=(
                            {"answer": final_letter, "support_refs": []}
                            if self.config.get("coseek1_direct_planner_answer", False)
                            else {}
                        ),
                        function_id="v11_direct_answer_from_letter_thought",
                    )
                ]
        if self.config.get("direct_answer_from_letter_thought", True):
            direct_letter = self.__extract_direct_answer_letter(thought, explicit_actions)
            if direct_letter:
                return [
                    Action(
                        function_name="answer",
                        parameters=(
                            {"answer": direct_letter, "support_refs": []}
                            if self.config.get("coseek1_direct_planner_answer", False)
                            else {}
                        ),
                        function_id="v11_direct_answer_from_letter_thought",
                    )
                ]
        actions = []

        try:
            response = call_llm_api(
                messages=[
                    {
                        "role": "user",
                        "content": f"Please call the appropriate tool(s) based on the following thought:\n{thought}",
                    }
                ],
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                tool_choice="required",
                tools=self.__planner_action_tools(),
                temperature=self.temperature,
            )

            message = response.choices[0].message.json()
            tool_calls = message.get("tool_calls", []) or []
            answer_call_id = None
            for tool_idx, tool_call in enumerate(tool_calls):
                function_name = tool_call.get("function", {}).get("name", None)
                parameters = json.loads(
                    tool_call.get("function", {}).get("arguments", "{}")
                )
                function_id = tool_call.get("id", None)
                if self.tool_registry.has_tool(function_name):
                    if function_name == "answer":
                        answer_call_id = tool_idx
                    actions.append(
                        Action(
                            function_name=function_name,
                            parameters=parameters,
                            function_id=function_id,
                        )
                    )
        except Exception as e:
            print(f"Error parsing actions: {e}")
            return explicit_actions

        actions = self.__normalize_coseek1_actions(actions)

        if len(actions) > 1 and answer_call_id is not None:
            actions.pop(answer_call_id)

        if self.__should_use_explicit_actions(actions, explicit_actions):
            return explicit_actions

        return actions

    def __compact_coseek1_thought(self, thought: str) -> str:
        if not self.config.get("coseek1_structured_planner", False):
            return thought
        if self.config.get("coseek1_planner_json_action", False):
            payload = extract_json_object(thought)
            if isinstance(payload, dict) and self.__extract_planner_json_actions(thought):
                return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        lines = [line.strip() for line in (thought or "").splitlines() if line.strip()]
        if not lines:
            return thought

        selected: dict[str, str] = {}
        for line in lines:
            match = re.match(r"(?is)^(STATE|NEXT_TOOL|WHY)\s*:\s*(.*)$", line)
            if not match:
                continue
            key = match.group(1).upper()
            if key not in selected:
                selected[key] = f"{key}: {match.group(2).strip()}"

        ordered = [selected[key] for key in ("STATE", "NEXT_TOOL", "WHY") if key in selected]
        return "\n".join(ordered) if ordered else thought

    def __extract_planner_json_actions(self, thought: str) -> List[Action]:
        payload = extract_json_object(thought)
        if not isinstance(payload, dict):
            return []

        action_items = []
        if isinstance(payload.get("actions"), list):
            action_items = payload.get("actions") or []
        elif isinstance(payload.get("action"), dict):
            action_items = [payload["action"]]
        elif any(key in payload for key in ("tool", "tool_name", "function", "name")):
            action_items = [payload]

        actions: List[Action] = []
        for idx, item in enumerate(action_items):
            if not isinstance(item, dict):
                continue
            tool_name = self.__planner_json_tool_name(item)
            if self.config.get("coseek1_structured_planner", False):
                tool_name = {"focus": "frame_verify", "skim": "skim_qwen"}.get(
                    tool_name,
                    tool_name,
                )
            if not tool_name or not self.tool_registry.has_tool(tool_name):
                continue
            if tool_name not in self.allowed_tool_names:
                continue
            params = self.__planner_json_parameters(item)
            if params is None:
                continue
            if tool_name == "answer" and self.config.get(
                "coseek1_direct_planner_answer", False
            ):
                params.setdefault("answer", item.get("answer"))
                params.setdefault("support_refs", item.get("support_refs") or [])
            if tool_name == "overview":
                params = {}
            elif tool_name == "answer":
                if self.config.get("coseek1_direct_planner_answer", False):
                    answer = str(params.get("answer") or "").strip().upper()
                    refs = params.get("support_refs") or []
                    if answer in set(self.config.get("_question_option_letters", "ABCD")):
                        params = {
                            "answer": answer,
                            "support_refs": [
                                str(ref).strip()
                                for ref in refs
                                if str(ref).strip()
                            ][:16]
                            if isinstance(refs, list)
                            else [],
                        }
                    else:
                        # Preserve the V40 answer path when a planner does not
                        # provide a valid direct answer payload.
                        params = {}
                else:
                    params = {}
            elif tool_name == "localize_qwen":
                if not params.get("search_windows"):
                    if "start_time" in params and "end_time" in params:
                        params["search_windows"] = [
                            [params["start_time"], params["end_time"]]
                        ]
                    else:
                        continue
            elif "start_time" not in params or "end_time" not in params:
                continue
            action_id = str(item.get("id") or f"coseek1_planner_json_{idx + 1}")
            actions.append(
                Action(
                    function_name=tool_name,
                    parameters=params,
                    function_id=action_id,
                )
            )
        return actions

    def __planner_json_tool_name(self, item: dict) -> str:
        function_value = item.get("function")
        if isinstance(function_value, dict):
            name = function_value.get("name")
            if name:
                return str(name).strip().lower()
        for key in ("tool", "tool_name", "name", "function"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return ""

    def __planner_json_parameters(self, item: dict) -> dict | None:
        params = (
            item.get("parameters")
            or item.get("params")
            or item.get("args")
            or item.get("arguments")
        )
        function_value = item.get("function")
        if params is None and isinstance(function_value, dict):
            params = function_value.get("arguments")

        if params is None:
            params = {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = extract_json_object(params) or {}
        if not isinstance(params, dict):
            return None

        params = dict(params)
        window = params.get("t_range") or params.get("window") or params.get("time_range")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            params.setdefault("start_time", window[0])
            params.setdefault("end_time", window[1])

        raw_windows = params.get("windows")
        if isinstance(raw_windows, list):
            normalized_windows = []
            for candidate in raw_windows[: max(
                1, int(self.config.get("focus_qwen_multi_window_max_windows") or 4)
            )]:
                if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
                    continue
                try:
                    candidate_start = max(0.0, float(candidate[0]))
                    candidate_end = min(float(candidate[1]), self.duration)
                except Exception:
                    continue
                if candidate_end <= candidate_start:
                    continue
                span = [candidate_start, candidate_end]
                if span not in normalized_windows:
                    normalized_windows.append(span)
            if normalized_windows:
                params["windows"] = normalized_windows
                params.setdefault(
                    "start_time", min(candidate[0] for candidate in normalized_windows)
                )
                params.setdefault(
                    "end_time", max(candidate[1] for candidate in normalized_windows)
                )
            else:
                params.pop("windows", None)

        raw_search_windows = params.get("search_windows")
        if isinstance(raw_search_windows, list):
            normalized_search_windows = []
            for candidate in raw_search_windows[: max(
                1, int(self.config.get("localize_qwen_max_search_windows") or 8)
            )]:
                if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
                    continue
                try:
                    candidate_start = max(0.0, float(candidate[0]))
                    candidate_end = min(float(candidate[1]), self.duration)
                except Exception:
                    continue
                if candidate_end <= candidate_start:
                    continue
                span = [candidate_start, candidate_end]
                if span not in normalized_search_windows:
                    normalized_search_windows.append(span)
            if normalized_search_windows:
                params["search_windows"] = normalized_search_windows
                params.setdefault(
                    "start_time", min(window[0] for window in normalized_search_windows)
                )
                params.setdefault(
                    "end_time", max(window[1] for window in normalized_search_windows)
                )
            else:
                params.pop("search_windows", None)

        for key in ("start_time", "end_time"):
            if key in params and params[key] is not None:
                try:
                    params[key] = float(params[key])
                except Exception:
                    return None
        if "start_time" in params and "end_time" in params:
            if float(params["end_time"]) <= float(params["start_time"]):
                return None

        if "mode" in params and params["mode"] is not None:
            params["mode"] = str(params["mode"])
        if "query" in params and params["query"] is not None:
            params["query"] = str(params["query"]).strip()
        if "localization_goal" in params and params["localization_goal"] is not None:
            params["localization_goal"] = str(params["localization_goal"]).strip()
        if "evidence_profile" in params and params["evidence_profile"] is not None:
            params["evidence_profile"] = str(params["evidence_profile"]).strip()
        if "top_k" in params:
            try:
                params["top_k"] = max(
                    1,
                    min(
                        int(params["top_k"]),
                        int(self.config.get("localize_qwen_max_top_k") or 4),
                    ),
                )
            except Exception:
                params.pop("top_k", None)
        return params

    def __normalize_coseek1_actions(self, actions: List[Action]) -> List[Action]:
        if not self.config.get("coseek1_structured_planner", False):
            return actions

        normalized: List[Action] = []
        for action in actions:
            name = action.function_name
            params = dict(action.parameters or {})
            function_id = action.function_id
            if name == "focus":
                name = "frame_verify"
            elif name == "skim":
                name = "skim_qwen"
            if name not in self.allowed_tool_names:
                continue
            if name in {"frame_verify", "focus_qwen", "skim_qwen"}:
                params.setdefault("query", self.question or "")
                params.setdefault("mode", "detail_verify" if name == "frame_verify" else "normal")
            elif name == "localize_qwen":
                params.setdefault(
                    "localization_goal",
                    params.get("query") or self.question or "",
                )
                params.setdefault("evidence_profile", "generic")
                params.setdefault("top_k", int(self.config.get("localize_qwen_top_k") or 3))
            if name == "focus_qwen" and self.__coseek1_focus_qwen_window_too_long(params):
                name = "skim_qwen"
                params["mode"] = "normal"
                function_id = f"{function_id or 'coseek1'}_long_focus_to_skim_qwen"
            normalized.append(
                Action(
                    function_name=name,
                    parameters=params,
                    function_id=function_id,
                )
            )
        return normalized

    def __coseek1_focus_qwen_window_too_long(self, params: dict) -> bool:
        windows = params.get("windows")
        if isinstance(windows, list) and windows:
            max_window_s = float(self.config.get("focus_qwen_max_window_s") or 20.0)
            valid_windows = []
            for window in windows:
                if not isinstance(window, (list, tuple)) or len(window) != 2:
                    continue
                try:
                    start_time = float(window[0])
                    end_time = float(window[1])
                except Exception:
                    continue
                if end_time > start_time:
                    valid_windows.append((start_time, end_time))
            if valid_windows:
                return any(
                    end_time - start_time > max_window_s
                    for start_time, end_time in valid_windows
                )
        try:
            start_time = float(params.get("start_time"))
            end_time = float(params.get("end_time"))
        except Exception:
            return False
        if end_time <= start_time:
            return False
        max_window_s = float(self.config.get("focus_qwen_max_window_s") or 20.0)
        return (end_time - start_time) > max_window_s

    def __extract_final_answer_letter(self, thought: str) -> str:
        lines = [line.strip() for line in (thought or "").splitlines() if line.strip()]
        if not lines:
            return ""
        stripped = (thought or "").strip().upper()
        if re.fullmatch(rf"[{re.escape(''.join(self.config.get('_question_option_letters', 'ABCD')))}]", stripped):
            return stripped
        last = lines[-1].strip().strip("`").strip().upper()
        if re.fullmatch(rf"[{re.escape(''.join(self.config.get('_question_option_letters', 'ABCD')))}]", last):
            return last
        answer_block = re.search(
            r"(?is)(?:^|\n)\s*(?:final\s+answer|answer)\s*:?\s*\n?\s*\(?([A-Z])\)?\s*$",
            thought or "",
        )
        if answer_block and answer_block.group(1).upper() in self.config.get('_question_option_letters', 'ABCD'):
            return answer_block.group(1).upper()
        return ""

    def __extract_direct_answer_letter(
        self,
        thought: str,
        explicit_actions: List[Action],
    ) -> str:
        if any(action.function_name != "answer" for action in explicit_actions):
            return ""
        lines = [line.strip() for line in (thought or "").splitlines() if line.strip()]
        if not lines:
            return ""
        stripped = (thought or "").strip().upper()
        if re.fullmatch(rf"[{re.escape(''.join(self.config.get('_question_option_letters', 'ABCD')))}]", stripped):
            return stripped
        last = lines[-1].strip().strip("`").strip().upper()
        if re.fullmatch(rf"[{re.escape(''.join(self.config.get('_question_option_letters', 'ABCD')))}]", last):
            return last
        final_match = re.search(
            r"(?is)\b(?:final\s+answer|answer|therefore|so)\s*(?:is|:)?\s*\(?([A-Z])\)?\s*$",
            thought or "",
        )
        if final_match and final_match.group(1).upper() in self.config.get('_question_option_letters', 'ABCD'):
            return final_match.group(1).upper()
        return ""

    def __should_use_explicit_actions(
        self,
        parsed_actions: List[Action],
        explicit_actions: List[Action],
    ) -> bool:
        if not self.config.get("prefer_explicit_thought_action", True):
            return False
        if not explicit_actions:
            return False
        explicit_has_observation = any(
            action.function_name != "answer" for action in explicit_actions
        )
        if not explicit_has_observation:
            return False
        if not parsed_actions:
            return True
        return all(action.function_name == "answer" for action in parsed_actions)

    def __extract_explicit_thought_actions(self, thought: str) -> List[Action]:
        """
        Recover explicit next-tool calls already written in the model thought.

        This is an action-consistency fallback for the two-stage agent loop:
        the planner may clearly write "Next action: Call focus ..." while the
        secondary tool parser returns answer().  We only read the explicit
        Next action / Tool Call region and only use this fallback when the
        secondary parser returns no observation action.
        """
        if not self.config.get("extract_explicit_thought_action", True):
            return []
        region = self.__explicit_action_region(thought or "")
        if not region:
            return []

        match = self.__first_explicit_tool_match(region)
        if not match:
            return []

        tool_name = match.group("tool").lower()
        if self.config.get("coseek1_structured_planner", False):
            tool_name = {"focus": "frame_verify", "skim": "skim_qwen"}.get(tool_name, tool_name)
        if not self.tool_registry.has_tool(tool_name):
            return []
        if tool_name not in self.allowed_tool_names:
            return []
        if tool_name == "answer":
            return [
                Action(
                    function_name="answer",
                    parameters={},
                    function_id="coseek1_explicit_answer",
                )
            ]

        parameters = self.__extract_action_parameters(tool_name, region[match.start() :])
        if parameters is None:
            return []

        return [
            Action(
                function_name=tool_name,
                parameters=parameters,
                function_id="v11_explicit_thought_action_1",
            )
        ]

    def __explicit_action_region(self, thought: str) -> str:
        if not thought:
            return ""
        next_action_matches = list(
            re.finditer(r"(?is)\bnext(?:\s+|_)(?:tool|action)\s*:", thought)
        )
        if next_action_matches:
            region = thought[next_action_matches[-1].start() :]
        else:
            tool_call = re.search(r"(?is)\btool\s*call\s*:", thought)
            if not tool_call:
                return ""
            region = thought[tool_call.start() :]

        # If the model hallucinated an observation after the planned call, do
        # not treat that synthetic/past observation as part of the next action.
        region = re.split(
            r"(?is)\bObservation\s+from\s+`|\bExecution\s+output\s+of\b",
            region,
            maxsplit=1,
        )[0]
        return region.strip()

    def __first_explicit_tool_match(self, text: str):
        names = set(self.allowed_tool_names)
        if self.config.get("coseek1_structured_planner", False):
            names.update({"skim", "focus"})
        tool_names = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        patterns = [
            rf"(?is)\btool\s*call\s*:?\s*`?(?P<tool>{tool_names})`?\b",
            rf"(?is)\bcall\s+(?:the\s+)?`?(?P<tool>{tool_names})`?(?:\s+tool)?\b",
            rf"(?is)\bi\s*(?:am|'m)\s+going\s+to\s+call\s+(?:the\s+)?`?(?P<tool>{tool_names})`?(?:\s+tool)?\b",
            rf"(?is)\b(?P<tool>{tool_names})\s*\(",
        ]
        matches = [m for pattern in patterns for m in re.finditer(pattern, text)]
        if not matches:
            return None
        return min(matches, key=lambda m: m.start())

    def __extract_action_parameters(self, tool_name: str, block: str):
        if tool_name in {"overview", "answer"}:
            return {}
        if tool_name not in {"skim", "skim_qwen", "focus", "focus_qwen", "frame_verify"}:
            return None

        time_range = self.__extract_time_range(block)
        if time_range is None:
            return None
        start_time, end_time = time_range
        if end_time <= start_time:
            return None

        mode = self.__extract_mode(block)
        if not mode:
            mode = "normal" if tool_name.startswith("skim") else "detail_verify"
        query = self.__extract_query(block)
        if not query:
            query = self.question or ""

        return {
            "query": query,
            "start_time": start_time,
            "end_time": end_time,
            "mode": mode,
        }

    def __extract_time_range(self, text: str):
        start_match = re.search(
            r"(?is)\bstart(?:_|\s*)time\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)\s*s?\b",
            text,
        )
        end_match = re.search(
            r"(?is)\bend(?:_|\s*)time\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)\s*s?\b",
            text,
        )
        if start_match and end_match:
            return (float(start_match.group(1)), float(end_match.group(1)))

        range_match = re.search(
            r"(?is)\b([0-9]+(?:\.[0-9]+)?)\s*s?\s*(?:-|–|—|to|~)\s*([0-9]+(?:\.[0-9]+)?)\s*s\b",
            text,
        )
        if range_match:
            return (float(range_match.group(1)), float(range_match.group(2)))
        return None

    def __extract_mode(self, text: str) -> str:
        mode_match = re.search(
            r"(?is)\bmode\s*[:=-]\s*[`'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]?",
            text,
        )
        if not mode_match:
            mode_match = re.search(
                r"(?is)\bwith\s+mode\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?",
                text,
            )
        return mode_match.group(1) if mode_match else ""

    def __extract_query(self, text: str) -> str:
        for quote, pattern in (
            ('"', r'(?is)\bquery(?:\s+to\s+\w+)?\s*[:=]\s*"((?:\\.|[^"\\])*)"'),
            ("'", r"(?is)\bquery(?:\s+to\s+\w+)?\s*[:=]\s*'((?:\\.|[^'\\])*)'"),
        ):
            quoted = re.search(pattern, text)
            if quoted:
                query = quoted.group(1).strip()
                return (
                    query.replace(f"\\{quote}", quote)
                    .replace('\\"', '"')
                    .replace("\\'", "'")
                )

        for line in text.splitlines():
            if not re.search(r"(?i)\bquery\b", line) or ":" not in line:
                continue
            query = line.split(":", 1)[1].strip()
            query = query.strip("`").strip()
            if query:
                return query
        return ""

    def __exec_action(self, action: Action) -> str:
        """
        Execute an action.
        """
        function_name = getattr(action, "function_name", None) if action else None
        parameters = dict(getattr(action, "parameters", {}) or {}) if action else {}

        if (
            self.config.get("coseek1_valid_overview_receipt_guard_enabled", False)
            and function_name == "overview"
            and has_valid_overview_receipt(self.observation_memory)
        ):
            action_signature = canonical_action_signature(function_name, parameters)
            record_blocked_overview(
                self.observation_memory,
                step=len(self.trajectory_steps) + 1,
                action_signature=action_signature,
            )
            return format_blocked_overview_feedback(self.observation_memory)

        if (
            function_name == "frame_verify"
            and self.config.get("coseek1_verified_observation_reuse_enabled", False)
        ):
            cached, scores = find_reusable_observation(
                getattr(self, "_verified_observation_cache", []),
                parameters=parameters,
                overlap_threshold=float(
                    self.config.get("coseek1_verified_observation_reuse_overlap") or 0.92
                ),
                query_similarity_threshold=float(
                    self.config.get(
                        "coseek1_verified_observation_reuse_query_similarity"
                    )
                    or 0.55
                ),
                context_pad_s=float(self.config.get("focus_context_pad_s") or 0.0),
                duration=float(self.duration or 0.0),
            )
            if cached is not None:
                return format_v10_observation(reused_payload(cached, scores=scores))

        if function_name == "answer":
            direct_answer = str(parameters.get("answer") or "").strip().upper()
            if (
                self.config.get("coseek1_direct_planner_answer", False)
                and direct_answer in set(self.config.get("_question_option_letters", "ABCD"))
            ):
                return direct_answer
            answer_messages = self.messages
            if self.config.get("coseek1_compact_answer_context", False):
                # execute_answer appends its evidence digest, so pass a fresh
                # list containing only stable context instead of mutating and
                # resending the complete local trajectory.
                answer_messages = [dict(self.messages[0]), dict(self.messages[1])]
            parameters = {
                "question": self.question,
                "messages": self.__subtitle_planner_messages(answer_messages),
                "memory": self.observation_memory,
            }
        else:
            parameters.update(
                {
                    "vr": self.vr,
                    "subtitles": self.subtitles,
                    "video_path": self.video_path,
                    "question": self.question,
                    "duration": self.duration,
                    "output_dir": self.output_dir,
                }
            )
            if (
                self.config.get("coseek1_tool_integrity_repair_enabled", False)
                and function_name in {"localize_qwen", "frame_verify"}
            ):
                # Internal-only handoff. The action exposed to the planner stays
                # unchanged; tools can reuse exact timestamps already in memory.
                parameters["memory"] = self.observation_memory
        
        if self.tool_registry.has_tool(function_name):
            tool_config = self.config
            if function_name == "answer" and self._adaptive_final_answer_request_active:
                tool_config = dict(self.config)
                tool_config["max_tokens"] = min(
                    int(self.max_tokens),
                    int(self._adaptive_token_budget.final_answer_max_tokens),
                )
            outcome = self.tool_registry.get_function(function_name)(
                config=tool_config, parameters=parameters
            )
            if outcome is None:
                outcome = "Tool execution failed."
            if (
                function_name == "frame_verify"
                and self.config.get("coseek1_verified_observation_reuse_enabled", False)
            ):
                payload = extract_v10_payload(outcome)
                if (
                    isinstance(payload, dict)
                    and payload.get("parse_ok") is not False
                    and payload.get("observer_backend") == "api"
                    and payload.get("observation_reused") is not True
                    and (
                        payload.get("timestamp_observations")
                        or payload.get("candidate_assessments")
                    )
                ):
                    cache = getattr(self, "_verified_observation_cache", None)
                    if cache is None:
                        cache = []
                        self._verified_observation_cache = cache
                    cache.append(
                        make_cache_entry(
                            cache_id=f"FVC{len(cache) + 1:04d}",
                            parameters=dict(getattr(action, "parameters", {}) or {}),
                            payload=payload,
                            context_pad_s=float(
                                self.config.get("focus_context_pad_s") or 0.0
                            ),
                            duration=float(self.duration or 0.0),
                        )
                    )
            return outcome
        else:
            raise ValueError(f"Invalid function name: {function_name}")

    def __attach_last_tool_observation_timing(
        self,
        *,
        tool_name: str,
        elapsed_s: float,
    ) -> None:
        observations = self.observation_memory.get("tool_observations") or []
        if not observations:
            return
        last = observations[-1]
        if isinstance(last, dict) and last.get("tool") == tool_name:
            last["elapsed_s"] = round(float(elapsed_s), 3)

    def __annotate_last_requirement_coverage(
        self,
        *,
        tool_name: str,
        parameters: dict | None,
        output: str,
    ) -> None:
        if not self.config.get("coseek1_requirement_coverage", False):
            return
        payload = extract_v10_payload(output)
        if not isinstance(payload, dict):
            return
        requirements = self.__query_requirement_specs()
        if not requirements:
            return
        coverage = self.__requirement_coverage_from_payload(
            payload,
            tool_name=tool_name,
            parameters=parameters or {},
            requirements=requirements,
        )
        if not coverage:
            return

        observations = self.observation_memory.get("tool_observations") or []
        if observations and isinstance(observations[-1], dict) and observations[-1].get("tool") == tool_name:
            observations[-1]["requirement_coverage"] = coverage
        scene_memory = self.observation_memory.get("scene_memory") or []
        if scene_memory and isinstance(scene_memory[-1], dict) and scene_memory[-1].get("source_tool") == tool_name:
            scene_memory[-1]["requirement_coverage"] = coverage

        if (
            tool_name == "frame_verify"
            and coverage.get("support_level") == "partial_verified"
            and coverage.get("missing_requirements")
        ):
            self.observation_memory.setdefault("open_gaps", []).append(
                {
                    "source_tool": tool_name,
                    "gap": (
                        "Verified evidence is partial: missing query requirements "
                        f"{coverage.get('missing_requirements')}"
                    ),
                    "suggested_window": coverage.get("suggested_window"),
                }
            )

    def __question_stem(self) -> str:
        text = self.question or ""
        return re.split(r"(?m)\n\s*\([A-Z]\)\s+", text, maxsplit=1)[0].strip()

    def __query_requirement_specs(self) -> list[dict[str, Any]]:
        stem = self.__question_stem()
        if not stem:
            return []

        relation_match = re.search(
            r"\b(while|when|during|as|after|before)\b",
            stem,
            flags=re.IGNORECASE,
        )
        has_attribute = bool(
            re.search(
                r"\b(color|colour|wearing|wears|dress|outfit|clothing|clothes)\b",
                stem,
                flags=re.IGNORECASE,
            )
        )
        if not relation_match and not has_attribute:
            return []

        specs: list[dict[str, Any]] = []
        if relation_match:
            before = stem[: relation_match.start()].strip(" ?.,")
            after = stem[relation_match.end() :].strip(" ?.,")
            target = self.__clean_target_requirement_phrase(before)
            context = self.__clean_context_requirement_phrase(after)
            if target:
                specs.append(self.__make_requirement("target_event", target, critical=True))
            if context:
                specs.append(self.__make_requirement("context_event", context, critical=True))

        if has_attribute:
            attribute = self.__extract_attribute_requirement(stem)
            if attribute:
                specs.append(self.__make_requirement("attribute", attribute, critical=True))

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for spec in specs:
            key = (str(spec.get("label")), str(spec.get("text")).lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
        return deduped

    def __clean_target_requirement_phrase(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(
            r"\b(?P<entity>woman|man|girl|boy|person|people|men|children|child)\b\s+"
            r"(?:who|that)\s+(?P<event>.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return self.__normalize_requirement_phrase(
                f"{match.group('entity')} {match.group('event')}"
            )
        match = re.search(
            r"\b(?P<entity>woman|man|girl|boy|person|people|men|children|child)\b(?P<tail>.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return self.__normalize_requirement_phrase(
                f"{match.group('entity')} {match.group('tail')}"
            )
        return self.__normalize_requirement_phrase(text)

    def __clean_context_requirement_phrase(self, text: str) -> str:
        text = re.split(r"\b(at|in|near|on)\s+the\s+beginning\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.sub(r"\bat\s+the\s+beginning\b", "", text, flags=re.IGNORECASE)
        return self.__normalize_requirement_phrase(text)

    def __extract_attribute_requirement(self, stem: str) -> str:
        if re.search(r"\bdress\b", stem, flags=re.IGNORECASE):
            return "dress color"
        if re.search(r"\boutfit|clothing|clothes|wear", stem, flags=re.IGNORECASE):
            return "outfit color"
        if re.search(r"\bcolor|colour\b", stem, flags=re.IGNORECASE):
            return "visible color"
        return ""

    def __normalize_requirement_phrase(self, text: str) -> str:
        text = re.sub(
            r"(?is)\b(what|which|who|where|when|why|how)\b.*?\b(is|are|was|were)\b",
            "",
            text,
        )
        text = re.sub(r"(?is)\b(the|a|an)\b", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" ?.,")

    def __make_requirement(self, label: str, text: str, *, critical: bool) -> dict[str, Any]:
        terms = self.__expanded_requirement_terms(text)
        threshold = 1 if label == "attribute" else min(2, max(1, len(terms)))
        if label == "context_event" and len(terms) >= 2:
            threshold = 2
        must_any_terms: set[str] = set()
        cooking_action_terms = {"cook", "cooks", "cooking", "cooked", "prepare", "preparing", "food", "apron", "stove"}
        entering_action_terms = {"enter", "enters", "entering", "entered", "arrive", "arrives"}
        phrase_terms = self.__text_terms(text)
        if label == "context_event" and phrase_terms & cooking_action_terms:
            must_any_terms = cooking_action_terms
        elif label == "target_event" and phrase_terms & entering_action_terms:
            must_any_terms = entering_action_terms | {"door", "doorway"}
        return {
            "label": label,
            "text": text,
            "terms": sorted(terms),
            "threshold": threshold,
            "critical": critical,
            "must_any_terms": sorted(must_any_terms),
        }

    def __expanded_requirement_terms(self, text: str) -> set[str]:
        base_terms = self.__text_terms(text)
        expanded = set(base_terms)
        synonym_groups = [
            {"enter", "enters", "entering", "entered", "door", "doorway", "arrive", "arrives", "walk", "walks"},
            {"cook", "cooks", "cooking", "cooked", "prepare", "preparing", "food", "kitchen", "apron", "stove"},
            {"man", "men", "male", "guy", "guys"},
            {"woman", "female", "lady", "girl"},
            {"dress", "outfit", "clothing", "clothes", "wear", "wearing", "top", "garment"},
            {"color", "colour", "black", "blue", "white", "green", "dark"},
        ]
        for group in synonym_groups:
            if expanded & group:
                expanded.update(group)
        return expanded

    def __text_terms(self, text: str | None) -> set[str]:
        stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "into", "onto",
            "what", "when", "where", "which", "who", "why", "how", "does", "did",
            "was", "were", "are", "is", "while", "during", "after", "before",
            "then", "there", "video", "question", "choice", "choices", "option",
            "options", "best", "given", "directly", "answer", "beginning",
            "near", "at", "in", "on", "of", "to", "or", "as",
        }
        terms: set[str] = set()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", (text or "").lower()):
            if token in stopwords:
                continue
            terms.add(token)
            if token.endswith("ing") and len(token) > 5:
                terms.add(token[:-3])
            if token.endswith("ed") and len(token) > 4:
                terms.add(token[:-2])
            if token.endswith("s") and len(token) > 4:
                terms.add(token[:-1])
        return terms

    def __requirement_coverage_from_payload(
        self,
        payload: dict,
        *,
        tool_name: str,
        parameters: dict,
        requirements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        positive_text = self.__payload_positive_text(payload)
        all_text = self.__payload_all_text(payload)
        positive_terms = self.__text_terms(positive_text)
        matched: list[str] = []
        missing: list[str] = []
        requirement_rows: list[dict[str, Any]] = []

        for req in requirements:
            label = str(req.get("label") or "")
            text = str(req.get("text") or "")
            terms = set(req.get("terms") or [])
            threshold = int(req.get("threshold") or 1)
            if label == "attribute" and payload.get("supports_options") and payload.get("detail_sufficient") is True:
                hit_count = max(threshold, len(terms & positive_terms))
                negative = False
            else:
                hit_count = len(terms & positive_terms)
                negative = self.__negative_requirement_mention(all_text, terms)

            must_any_terms = set(req.get("must_any_terms") or [])
            has_must_term = not must_any_terms or bool(must_any_terms & positive_terms)
            if (
                tool_name == "frame_verify"
                and payload.get("detail_sufficient") is True
                and payload.get("supports_options")
                and label in {"target_event", "context_event"}
                and hit_count >= threshold
                and has_must_term
            ):
                negative = False
            is_matched = bool(hit_count >= threshold and has_must_term and not negative)
            row = {
                "label": label,
                "text": text,
                "matched": is_matched,
                "hit_terms": sorted(terms & positive_terms),
                "hit_must_terms": sorted(must_any_terms & positive_terms),
                "negative_mention": negative,
            }
            requirement_rows.append(row)
            if is_matched:
                matched.append(text)
            else:
                missing.append(text)

        critical_missing = [
            row["text"]
            for row, req in zip(requirement_rows, requirements)
            if req.get("critical") and not row.get("matched")
        ]
        supports = payload.get("supports_options") or []
        relation_verified = not critical_missing
        support_level = "none"
        if supports and tool_name == "frame_verify":
            support_level = "verified" if relation_verified else "partial_verified"
        elif supports:
            support_level = "routing"

        return {
            "matched_requirements": matched,
            "missing_requirements": missing,
            "critical_missing_requirements": critical_missing,
            "requirement_rows": requirement_rows,
            "relation_verified": relation_verified,
            "support_level": support_level,
            "support_scope": self.__support_scope_summary(payload, matched, missing),
            "suggested_window": payload.get("suggest_focus_window"),
        }

    def __payload_positive_text(self, payload: dict) -> str:
        parts: list[str] = []
        for key in (
            "global_summary",
            "overall_summary",
            "observed_event",
            "local_motion",
            "interaction",
            "state_change",
        ):
            value = payload.get(key)
            if value:
                parts.append(str(value))
        for item in payload.get("timestamp_observations") or []:
            if isinstance(item, dict):
                parts.append(str(item.get("description") or ""))
                parts.extend(str(tag) for tag in item.get("event_tags") or [])
        for item in payload.get("scene_summaries") or []:
            if isinstance(item, dict):
                parts.append(str(item.get("summary") or ""))
        parts.extend(str(x) for x in payload.get("supports_options") or [])
        return " ".join(part for part in parts if part)

    def __payload_all_text(self, payload: dict) -> str:
        parts = [self.__payload_positive_text(payload)]
        for key in ("missing_detail", "reason", "rationale"):
            value = payload.get(key)
            if value:
                parts.append(str(value))
        for item in payload.get("scene_summaries") or []:
            if isinstance(item, dict):
                parts.append(str(item.get("missing_detail") or ""))
        return " ".join(part for part in parts if part)

    def __negative_requirement_mention(self, text: str, terms: set[str]) -> bool:
        if not text or not terms:
            return False
        lower = text.lower()
        useful_terms = [re.escape(term) for term in terms if len(term) >= 4]
        if not useful_terms:
            return False
        term_re = r"(?:" + "|".join(useful_terms) + r")"
        patterns = [
            rf"\b(no|not|without|neither|cannot|can't|unclear|missing)\b[^.。;]{{0,80}}\b{term_re}\b",
            rf"\b{term_re}\b[^.。;]{{0,80}}\b(not visible|not clearly visible|no clear|unclear|cannot confirm|can't confirm)\b",
        ]
        return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in patterns)

    def __support_scope_summary(
        self,
        payload: dict,
        matched: list[str],
        missing: list[str],
    ) -> str:
        summary = str(payload.get("overall_summary") or payload.get("observed_event") or "").strip()
        if not summary:
            summary = "verified local window"
        if missing:
            return f"{summary}; missing {', '.join(missing)}"
        return summary

    def __has_observation_memory(self) -> bool:
        return any(
            self.observation_memory.get(key)
            for key in ("timestamped_observations", "scene_memory", "tool_observations")
        )

    def __force_initial_observation_if_needed(
        self,
        actions: List[Action],
        *,
        step: int,
    ) -> List[Action]:
        if self.__has_observation_memory():
            return actions
        if not actions or actions[0].function_name == "answer":
            return [
                Action(
                    function_name="overview",
                    parameters={},
                    function_id=f"v10_forced_overview_{step + 1}",
                )
            ]
        return actions

    def __scene_has_skim(self, scene_id: str) -> bool:
        for item in self.observation_memory.get("scene_memory") or []:
            if not isinstance(item, dict):
                continue
            if item.get("scene_id") != scene_id:
                continue
            if item.get("source_tool") in {"skim", "skim_qwen"}:
                return True
        return False

    def __auto_prepend_scene_skim_before_focus(
        self,
        actions: List[Action],
        *,
        step: int,
    ) -> List[Action]:
        if not actions or not self.config.get("require_scene_skim_before_focus", True):
            return actions

        out: List[Action] = []
        inserted_for_scene: set[str] = set()
        for action in actions:
            if action.function_name != "focus":
                out.append(action)
                continue

            params = dict(action.parameters or {})
            try:
                start_time = float(params.get("start_time"))
                end_time = float(params.get("end_time"))
            except Exception:
                out.append(action)
                continue

            scene_context, _cache_path = get_scene_context_for_window(
                self.config,
                video_path=self.video_path,
                duration_s=self.duration,
                start_time=start_time,
                end_time=end_time,
            )
            scene_id = str((scene_context or {}).get("scene_id") or "")
            if not scene_id or scene_id in inserted_for_scene or self.__scene_has_skim(scene_id):
                out.append(action)
                continue

            scene_start, scene_end = scene_context.get("t_range") or [start_time, end_time]
            scene_start = float(scene_start)
            scene_end = float(scene_end)
            context_s = float(self.config.get("pre_focus_skim_context_s") or 10.0)
            max_window_s = float(self.config.get("pre_focus_skim_max_window_s") or 45.0)
            skim_start = max(scene_start, start_time - context_s)
            skim_end = min(scene_end, end_time + context_s)
            if skim_end - skim_start > max_window_s:
                midpoint = (start_time + end_time) / 2.0
                skim_start = max(scene_start, midpoint - max_window_s / 2.0)
                skim_end = min(scene_end, midpoint + max_window_s / 2.0)
            if skim_end <= skim_start:
                out.append(action)
                continue

            query = params.get("query") or self.question or ""
            out.append(
                Action(
                    function_name="skim",
                    parameters={
                        "query": (
                            f"Scene-level skim before fine focus. {query} "
                            "Find the evidence-bearing moment and suggest the best tight focus window."
                        ),
                        "start_time": round(float(skim_start), 1),
                        "end_time": round(float(skim_end), 1),
                        "mode": "normal",
                    },
                    function_id=f"v11_auto_prefocus_skim_{step + 1}_{len(out) + 1}",
                )
            )
            inserted_for_scene.add(scene_id)
            out.append(action)
        return out

    def __auto_probe_scene_before_answer(
        self,
        actions: List[Action],
        *,
        step: int,
    ) -> List[Action]:
        if not actions or not self.config.get("query_aware_scene_probe", False):
            return actions
        if not all(action.function_name == "answer" for action in actions):
            return actions
        max_count = int(self.config.get("query_aware_scene_probe_max_per_question") or 0)
        if max_count <= 0 or self._query_scene_probe_count >= max_count:
            return actions

        candidate = self.__select_query_probe_candidate()
        if candidate is None:
            return actions
        scene_id, start_time, end_time, summary, missing = candidate
        self._query_scene_probe_count += 1
        self._query_scene_probe_scene_ids.add(scene_id)
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages[-1]["content"] = (
                "The planner attempted to answer before all candidate scene evidence was checked. "
                "The router deferred the answer and scheduled a query-aware scene probe instead."
            )
        return [
            Action(
                function_name="skim",
                parameters={
                    "query": (
                        "Query-aware scene probe before answering. "
                        f"Question and choices: {self.question or ''}\n"
                        f"Candidate scene {scene_id} summary: {summary}\n"
                        f"Missing detail from overview: {missing}\n"
                        "Inspect this scene for direct visual concepts named in the question and choices. "
                        "Report whether the scene contains the target evidence, what is missing, "
                        "and the best tight focus window for API verification."
                    ),
                        "start_time": round(float(start_time), 1),
                        "end_time": round(float(end_time), 1),
                        "mode": "normal",
                        "query_aware_probe": True,
                    },
                    function_id=f"v13_query_scene_probe_{step + 1}_{self._query_scene_probe_count}",
                )
        ]

    def __select_query_probe_candidate(self):
        scenes = [
            item
            for item in self.observation_memory.get("scene_memory") or []
            if isinstance(item, dict)
        ]
        if not scenes:
            return None

        def as_range(value):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return None
            try:
                start = max(0.0, float(value[0]))
                end = min(float(value[1]), self.duration)
            except Exception:
                return None
            if end <= start:
                return None
            return start, end

        def truthy(value) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes", "1", "possible", "likely"}
            return bool(value)

        def safe_float(value, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "into", "onto",
            "what", "when", "where", "which", "who", "why", "how", "does", "did",
            "was", "were", "are", "after", "before", "then", "there", "video",
            "question", "choice", "choices", "option", "options",
        }
        query_terms = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", self.question or "")
            if token.lower() not in stopwords
            for token in [token.lower()]
        }

        candidates = []
        inspected_scenes = {
            str(item.get("scene_id") or "")
            for item in scenes
            if item.get("source_tool") in {"skim", "skim_qwen", "focus", "focus_qwen"}
        }
        inspected_scenes.update(self._query_scene_probe_scene_ids)
        for idx, item in enumerate(scenes):
            source = item.get("source_tool")
            if source != "overview":
                continue
            scene_id = str(item.get("scene_id") or "")
            if not scene_id or scene_id in inspected_scenes:
                continue
            focus_windows = item.get("suggest_focus_windows") or []
            max_window_s = float(self.config.get("query_aware_scene_probe_max_window_s") or 90.0)
            scene_window = as_range(item.get("t_range"))
            focus_ranges = []
            for focus in focus_windows:
                focus_range = as_range(focus)
                if focus_range is not None:
                    focus_ranges.append(focus_range)
            window = None
            if (
                self.config.get("query_aware_scene_probe_prefer_scene_range", True)
                and scene_window is not None
                and scene_window[1] - scene_window[0] <= max_window_s
            ):
                window = scene_window
            elif focus_ranges:
                focus_start = min(start for start, _end in focus_ranges)
                focus_end = max(end for _start, end in focus_ranges)
                if scene_window is not None:
                    pad_s = min(5.0, max(0.0, (scene_window[1] - scene_window[0]) * 0.1))
                    focus_start = max(scene_window[0], focus_start - pad_s)
                    focus_end = min(scene_window[1], focus_end + pad_s)
                window = (focus_start, focus_end)
            if window is None:
                window = scene_window
            if window is None:
                continue

            start, end = window
            if end - start > max_window_s:
                midpoint = (start + end) / 2.0
                start = max(0.0, midpoint - max_window_s / 2.0)
                end = min(self.duration, midpoint + max_window_s / 2.0)
            if end <= start:
                continue

            possible = item.get("possible_evidence")
            possible_score = 1 if truthy(possible) else 0
            has_focus = 1 if focus_windows else 0
            summary = str(item.get("summary") or "").strip()
            missing = str(item.get("missing_detail") or "").strip()
            text_terms = set(
                re.findall(
                    r"[a-zA-Z][a-zA-Z0-9_'-]{2,}",
                    f"{scene_id} {summary}".lower(),
                )
            )
            lexical_hits = len(query_terms & text_terms)
            missing_score = 1 if missing else 0
            relevance_score = safe_float(item.get("relevance"), 0.0)
            min_relevance = float(
                self.config.get("query_aware_scene_probe_api_relevance_threshold") or 0.3
            )
            if (
                possible_score <= 0
                and has_focus <= 0
                and lexical_hits <= 0
                and relevance_score < min_relevance
            ):
                continue
            span_score = -(end - start)
            order_score = -idx
            score = (
                possible_score,
                has_focus,
                relevance_score,
                lexical_hits,
                missing_score,
                span_score,
                order_score,
            )
            candidates.append((score, scene_id, start, end, summary, missing))

        if not candidates:
            return None
        _score, scene_id, start, end, summary, missing = max(candidates, key=lambda item: item[0])
        return scene_id, start, end, summary, missing

    def __config_list(self, key: str, default: str = "") -> set[str]:
        raw = self.config.get(key)
        if raw is None:
            raw = default
        if isinstance(raw, (list, tuple, set)):
            return {str(item).strip() for item in raw if str(item).strip()}
        return {chunk.strip() for chunk in str(raw).split(",") if chunk.strip()}

    def __qwen_first_enabled(self) -> bool:
        return bool(self.config.get("qwen_first_observer", False))

    def __qwen_first_route_actions(
        self,
        actions: List[Action],
        *,
        step: int,
    ) -> List[Action]:
        if not actions or not self.__qwen_first_enabled():
            return actions

        out: List[Action] = []
        for idx, action in enumerate(actions, start=1):
            routed = self.__qwen_first_route_action(action, step=step, index=idx)
            out.append(routed)
        return out

    def __qwen_first_route_action(
        self,
        action: Action,
        *,
        step: int,
        index: int,
    ) -> Action:
        if action.function_name not in {"skim", "focus"}:
            return action
        params = dict(action.parameters or {})
        try:
            start_time = float(params.get("start_time"))
            end_time = float(params.get("end_time"))
        except Exception:
            return action
        if end_time <= start_time:
            return action

        mode = str(params.get("mode") or ("normal" if action.function_name == "skim" else "detail_verify"))
        window_s = end_time - start_time
        if action.function_name == "skim":
            if not self.config.get("qwen_first_skim", True):
                return action
            if self.__qwen_first_repeat_window_should_escalate(action):
                return action
            allowed_modes = self.__config_list(
                "qwen_first_skim_modes",
                "normal,option_coverage,alternative_window",
            )
            max_window_s = float(self.config.get("qwen_first_skim_max_window_s") or 90.0)
            qwen_tool = "skim_qwen"
        else:
            if not self.config.get("qwen_first_focus", True):
                return action
            if self.__qwen_first_repeat_window_should_escalate(action):
                return action
            allowed_modes = self.__config_list(
                "qwen_first_focus_modes",
                "detail_verify,option_verify,wider_context",
            )
            max_window_s = float(self.config.get("qwen_first_focus_max_window_s") or 20.0)
            qwen_tool = "focus_qwen"

        if allowed_modes and mode not in allowed_modes:
            return action
        if window_s > max_window_s:
            return action

        source_id = action.function_id or f"v12_qwen_first_source_{step + 1}_{index}"
        qwen_id = f"{source_id}_qwen_first"
        api_id = f"{source_id}_api_after_qwen"
        self._qwen_first_api_fallbacks[qwen_id] = Action(
            function_name=action.function_name,
            parameters=params,
            function_id=api_id,
        )
        return Action(
            function_name=qwen_tool,
            parameters=params,
            function_id=qwen_id,
        )

    def __qwen_first_repeat_window_should_escalate(self, action: Action) -> bool:
        if not self.config.get("qwen_first_repeat_window_escalates_to_api", True):
            return False
        params = dict(action.parameters or {})
        try:
            start_time = float(params.get("start_time"))
            end_time = float(params.get("end_time"))
        except Exception:
            return False
        if end_time <= start_time:
            return False

        qwen_tool = f"{action.function_name}_qwen"
        for item in reversed(self.observation_memory.get("tool_observations") or []):
            if not isinstance(item, dict):
                continue
            if item.get("tool") != qwen_tool:
                continue
            if item.get("observer_backend") != "local_qwen":
                continue
            old_params = item.get("parameters") or {}
            try:
                old_start = float(old_params.get("start_time"))
                old_end = float(old_params.get("end_time"))
            except Exception:
                continue
            overlap = min(end_time, old_end) - max(start_time, old_start)
            if overlap <= 0:
                continue
            union = max(end_time, old_end) - min(start_time, old_start)
            if union > 0 and overlap / union >= 0.5:
                return True
        return False

    def __route_repeated_skim_for_information_gain(
        self,
        actions: List[Action],
        *,
        step: int,
        thought: str,
    ) -> tuple[List[Action], str]:
        if not self.config.get("coseek1_visited_window_information_gain", False):
            return actions, thought
        if not actions:
            return actions, thought

        routed: list[Action] = []
        changed_action: Action | None = None
        changed_reason = ""
        overlap_threshold = float(
            self.config.get("coseek1_visited_window_overlap") or 0.85
        )
        for action in actions:
            if action.function_name != "skim_qwen":
                routed.append(action)
                continue
            params = dict(action.parameters or {})
            rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
            if rng is None or not self.__has_overlapping_tool_window(
                {"skim_qwen"},
                rng,
                threshold=overlap_threshold,
            ):
                routed.append(action)
                continue

            query = str(params.get("query") or self.question or "Inspect visual evidence.")
            candidate_window = self.__select_memory_suggested_focus_window(
                rng,
                max_window_s=float(self.config.get("focus_qwen_max_window_s") or 20.0),
            )
            if candidate_window is not None:
                already_focused = self.__has_overlapping_tool_window(
                    {"focus_qwen"},
                    candidate_window,
                    threshold=0.7,
                )
                tool_name = "frame_verify" if already_focused else "focus_qwen"
                replacement = Action(
                    function_name=tool_name,
                    parameters={
                        "query": query,
                        "start_time": round(candidate_window[0], 1),
                        "end_time": round(candidate_window[1], 1),
                        "mode": "option_verify" if already_focused else "normal",
                    },
                    function_id=f"v20_information_gain_{tool_name}_{step + 1}",
                )
                routed.append(replacement)
                changed_action = replacement
                changed_reason = (
                    "The requested skim window is already represented in trajectory memory; "
                    "inspect its best unverified candidate window instead of resampling identical frames."
                )
                continue

            alternative = self.__select_unverified_overview_candidate()
            if alternative is not None:
                replacement = self.__action_for_candidate_scene(alternative, step=step)
                if replacement is not None:
                    routed.append(replacement)
                    changed_action = replacement
                    changed_reason = (
                        "The requested skim window has already been searched without a new focus candidate; "
                        "move to the next unvisited overview candidate."
                    )
                    continue

            center_window = self.__center_crop_range(
                rng,
                max_window_s=float(self.config.get("focus_qwen_max_window_s") or 20.0),
            )
            if not self.__has_overlapping_tool_window(
                {"frame_verify"},
                center_window,
                threshold=0.7,
            ):
                replacement = Action(
                    function_name="frame_verify",
                    parameters={
                        "query": query,
                        "start_time": round(center_window[0], 1),
                        "end_time": round(center_window[1], 1),
                        "mode": "option_verify",
                    },
                    function_id=f"v20_information_gain_frame_verify_{step + 1}",
                )
                routed.append(replacement)
                changed_action = replacement
                changed_reason = (
                    "No unvisited coarse candidate remains in this window; use a stronger observer "
                    "on a bounded subwindow rather than repeating the same skim."
                )
                continue

            if "defer_answer" in str(action.function_id or ""):
                replacement = Action(
                    function_name="answer",
                    parameters={},
                    function_id=f"v20_information_gain_resume_answer_{step + 1}",
                )
                routed.append(replacement)
                changed_action = replacement
                changed_reason = (
                    "The deferred search has exhausted its already-inspected window and produced no "
                    "new candidate; answer from accumulated evidence instead of looping."
                )
                continue

            routed.append(action)

        if changed_action is None:
            return routed, thought
        new_thought = self.__format_router_thought(
            changed_action,
            state="Trajectory memory detected an unchanged revisit with no information gain.",
            why=changed_reason,
        )
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages[-1]["content"] = new_thought
        return routed, new_thought

    def __qwen_first_should_verify_with_api(
        self,
        action: Action,
        outcome: str,
    ) -> bool:
        if not self.__qwen_first_enabled():
            return False
        if action.function_name not in {"focus_qwen", "skim_qwen"}:
            return False
        if action.function_id not in self._qwen_first_api_fallbacks:
            return False

        payload = extract_v10_payload(outcome)
        if not isinstance(payload, dict):
            return bool(self.config.get("qwen_first_verify_parse_error", True))

        backend = str(payload.get("observer_backend") or "")
        if "error" in backend or "fallback_after_local_qwen_error" in backend:
            return bool(self.config.get("qwen_first_verify_parse_error", True))
        if payload.get("parse_ok") is False:
            return bool(self.config.get("qwen_first_verify_parse_error", True))

        if action.function_name == "focus_qwen":
            if not self.config.get("qwen_first_verify_uncertain_focus", False):
                return False
            if bool(payload.get("detail_sufficient")):
                return False
            missing = str(payload.get("missing_detail") or "").strip()
            return bool(missing)

        params = action.parameters or {}
        if bool(params.get("query_aware_probe")) and self.config.get(
            "query_aware_scene_probe_api_verify", True
        ):
            relevance = payload.get("relevance")
            try:
                relevance_f = float(relevance)
            except Exception:
                relevance_f = 0.0
            min_relevance = float(
                self.config.get("query_aware_scene_probe_api_relevance_threshold") or 0.3
            )
            possible = payload.get("possible_evidence")
            if possible is None:
                possible = payload.get("contains_evidence")
            if bool(possible):
                return True
            if relevance_f >= min_relevance:
                return True
            missing = str(payload.get("missing_detail") or "").strip()
            has_timestamps = bool(payload.get("timestamp_observations"))
            has_event = bool(str(payload.get("observed_event") or "").strip())
            return bool(missing and (has_timestamps or has_event))

        # Skim is used for routing. Escalate only when Qwen could not produce a
        # usable coarse observation, or when configured to verify ambiguous hits.
        if not self.config.get("qwen_first_verify_uncertain_skim", False):
            return False
        relevance = payload.get("relevance")
        try:
            relevance_f = float(relevance)
        except Exception:
            relevance_f = 0.0
        has_timestamps = bool(payload.get("timestamp_observations"))
        has_event = bool(str(payload.get("observed_event") or "").strip())
        if not has_timestamps and not has_event:
            return True
        min_relevance = float(self.config.get("qwen_first_skim_api_relevance_threshold") or 0.2)
        return relevance_f < min_relevance and bool(payload.get("missing_detail"))

    def __append_tool_call_message(self, action: Action) -> None:
        assistant_message = None
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                assistant_message = message
                break
        if assistant_message is None:
            return
        assistant_message.setdefault("tool_calls", [])
        assistant_message["tool_calls"].append(
            {
                "id": action.function_id,
                "type": "function",
                "function": {
                    "name": action.function_name,
                    "arguments": str(action.parameters),
                },
            }
        )

    def __inline_localize_verify_action(
        self,
        *,
        localize_action: Action,
        localize_output: str,
        step: int,
    ) -> Action | None:
        """Build one multi-window API verification child for a useful localization."""
        if not self.config.get("coseek1_localize_inline_verify", False):
            return None
        if localize_action.function_name != "localize_qwen":
            return None
        payload = extract_v10_payload(localize_output)
        if not isinstance(payload, dict) or payload.get("parse_ok") is False:
            return None
        novelty_enabled = bool(
            self.config.get("localize_qwen_candidate_novelty_enabled", False)
        )
        if novelty_enabled and payload.get("localization_progress") == "no_new_information":
            return None

        max_windows = max(1, int(self.config.get("localize_inline_verify_max_windows") or 3))
        candidates = []
        for item in payload.get("ranked_candidates") or []:
            if not isinstance(item, dict):
                continue
            if novelty_enabled and item.get("novelty_status") == "repeated":
                continue
            if not (
                item.get("verification_eligible") is True
                or str(item.get("localization_status") or "").lower() == "found"
            ):
                continue
            window = item.get("recommended_verify_window") or item.get("t_range")
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                continue
            try:
                start, end = float(window[0]), float(window[1])
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            candidates.append(
                {
                    "candidate_id": str(item.get("candidate_id") or f"LQ{len(candidates) + 1:03d}"),
                    "rank": int(item.get("rank") or len(candidates) + 1),
                    "source_index": int(item.get("source_index") or 0),
                    "t_range": [round(start, 3), round(end, 3)],
                    "summary": str(item.get("fine_summary") or "").strip()[:480],
                    "selection_class": str(
                        item.get("selection_class") or "ambiguous"
                    ),
                    "mandatory_positive": item.get("mandatory_positive") is True,
                    "timestamp_anchors": list(item.get("timestamp_anchors") or []),
                    "localized_timestamp_anchors": list(
                        item.get("localized_timestamp_anchors") or []
                    ),
                    "memory_timestamp_anchors": list(
                        item.get("memory_timestamp_anchors") or []
                    ),
                    "localized_evidence_anchors": list(
                        item.get("localized_evidence_anchors") or []
                    ),
                    "source_search_window": list(
                        item.get("source_search_window") or item.get("t_range") or []
                    ),
                    "localized_verify_window": list(
                        item.get("localized_verify_window") or window
                    ),
                }
            )
            if len(candidates) >= max_windows:
                break
        event_candidates = self.__event_coverage_inline_candidates()
        # P123: event coverage provides semantic labels, but the current localize
        # result owns the pixels selected for immediate verification. Replacing
        # fresh LQ windows with older EVT anchors can discard a newly localized
        # event or verify stale context-only evidence.
        if not candidates:
            return None

        profile = str(localize_action.parameters.get("evidence_profile") or "generic")
        mode = (
            "temporal_strip"
            if profile in {"event_boundary", "state_change", "temporal_order"}
            else "option_verify"
        )
        if len(event_candidates) >= 2:
            profile = "temporal_order"
            event_labels = "; ".join(
                f"{item['candidate_id']}: {item['event']}" for item in event_candidates
            )
            retrieval_hint = (
                "For each candidate window, neutrally identify which of these events, if any, "
                f"is visibly shown: {event_labels}. Report timestamped facts for every window; "
                "do not assume an event is present and do not infer chronology from window order."
            )
        else:
            retrieval_hint = str(
                payload.get("suggest_frame_verify_query")
                or localize_action.parameters.get("localization_goal")
                or self.question
                or ""
            ).strip()
        semantic_decoupling_enabled = bool(
            self.config.get(
                "coseek1_retrieval_verify_semantic_decoupling_enabled", False
            )
        )
        if semantic_decoupling_enabled:
            query = str(self.question or "").strip() or retrieval_hint
        else:
            query = retrieval_hint
        start = min(item["t_range"][0] for item in candidates)
        end = max(item["t_range"][1] for item in candidates)
        verify_parameters = {
            "query": query,
            "start_time": start,
            "end_time": end,
            "mode": mode,
            "candidate_windows": candidates,
            "windows": [item["t_range"] for item in candidates],
            "inline_from_localize": True,
            "inline_event_coverage": len(event_candidates) >= 2,
            "event_coverage_max_windows": len(event_candidates),
            "source_localize_action_id": localize_action.function_id,
        }
        if semantic_decoupling_enabled:
            verify_parameters.update(
                {
                    "retrieval_hint": retrieval_hint,
                    "verification_objective_source": "original_question",
                }
            )
        return Action(
            function_name="frame_verify",
            parameters=verify_parameters,
            function_id=f"v46_inline_verify_{step + 1}_{localize_action.function_id}",
        )

    def __event_coverage_inline_candidates(self) -> list[dict[str, Any]]:
        """Convert existing event anchors into one bounded verify batch."""
        if not self.config.get("coseek1_event_coverage_inline_verify", False):
            return []
        coverage = build_event_coverage(
            self.observation_memory,
            question=self.question,
            max_items=int(self.config.get("coseek1_compact_event_coverage_max_items") or 8),
        )
        if len(coverage) < 2:
            return []

        max_windows = max(
            2,
            int(self.config.get("coseek1_event_coverage_inline_verify_max_windows") or 4),
        )
        radius_s = max(
            1.0,
            float(self.config.get("coseek1_event_coverage_inline_verify_radius_s") or 5.0),
        )
        candidates: list[dict[str, Any]] = []
        for item in coverage:
            timestamp = item.get("earliest_timestamp_s")
            if not isinstance(timestamp, (int, float)):
                continue
            start = max(0.0, float(timestamp) - radius_s)
            end = min(float(self.duration), float(timestamp) + radius_s)
            if end <= start:
                continue
            candidates.append(
                {
                    "candidate_id": str(item.get("event_id") or f"EVT{len(candidates) + 1:02d}"),
                    "rank": len(candidates) + 1,
                    "event": str(item.get("event") or "event")[:160],
                    "t_range": [round(start, 3), round(end, 3)],
                    "summary": str(item.get("earliest_fact") or item.get("fact") or "")[:360],
                    "source": "compact_event_coverage",
                    "source_evidence_ref": item.get("earliest_evidence_ref"),
                    "source_evidence_level": item.get("earliest_status"),
                }
            )
            if len(candidates) >= max_windows:
                break
        return candidates

    def __defer_answer_if_evidence_insufficient(
        self,
        actions: List[Action],
        *,
        step: int,
        thought: str,
    ) -> tuple[List[Action], str]:
        if not self.config.get("coseek1_defer_insufficient_answer", True):
            return actions, thought
        if not actions or not all(action.function_name == "answer" for action in actions):
            return actions, thought
        if step + 1 >= int(self.max_steps):
            return actions, thought
        partial = self.__latest_frame_verify_partial_requirement_coverage()
        if partial is not None and self.config.get(
            "coseek1_defer_partial_requirement_answer",
            False,
        ):
            _latest_item, coverage = partial
            candidate = self.__select_requirement_recovery_candidate(coverage)
            if candidate is not None:
                action = self.__action_for_requirement_candidate(
                    candidate,
                    coverage,
                    step=step,
                )
                if action is not None:
                    new_thought = self.__format_router_thought(
                        action,
                        state=(
                            "The latest frame_verify supports an option only for a local "
                            "attribute, but the same verified window is missing required "
                            f"query context: {coverage.get('critical_missing_requirements') or coverage.get('missing_requirements')}."
                        ),
                        why=(
                            "Answer is deferred because the evidence must cover the target "
                            "event and its context in the same window; verify the best "
                            "candidate window that covers the missing requirement."
                        ),
                    )
                    if self.messages and self.messages[-1].get("role") == "assistant":
                        self.messages[-1]["content"] = new_thought
                    return [action], new_thought
        if not self.__latest_frame_verify_is_insufficient():
            return actions, thought

        candidate = self.__select_unverified_overview_candidate()
        if candidate is None:
            return actions, thought

        action = self.__action_for_candidate_scene(candidate, step=step)
        if action is None:
            return actions, thought

        scene_id = str(candidate.get("scene_id") or "candidate")
        t_range = candidate.get("t_range") or []
        new_thought = (
            "STATE: The latest frame_verify is not sufficient for a final answer "
            "because direct support is missing or key query context remains unresolved; "
            f"unverified candidate scene {scene_id} {t_range} remains.\n"
            f"NEXT_TOOL: {self.__format_action_as_next_tool(action)}\n"
            "WHY: Answer is deferred because evidence is incomplete; inspect the "
            "remaining possible-evidence scene before choosing an option."
        )
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages[-1]["content"] = new_thought
        return [action], new_thought

    def __latest_frame_verify_partial_requirement_coverage(self):
        for item in reversed(self.observation_memory.get("tool_observations") or []):
            if not isinstance(item, dict):
                continue
            if item.get("tool") != "frame_verify":
                continue
            coverage = item.get("requirement_coverage")
            if not isinstance(coverage, dict):
                return None
            if coverage.get("support_level") != "partial_verified":
                return None
            missing = coverage.get("critical_missing_requirements") or coverage.get("missing_requirements") or []
            if not missing:
                return None
            return item, coverage
        return None

    def __select_requirement_recovery_candidate(self, coverage: dict):
        requirements = self.__query_requirement_specs()
        if not requirements:
            return None
        missing_texts = {
            str(item).lower()
            for item in (
                coverage.get("critical_missing_requirements")
                or coverage.get("missing_requirements")
                or []
            )
        }
        missing_requirements = [
            req for req in requirements if str(req.get("text") or "").lower() in missing_texts
        ]
        if not missing_requirements:
            missing_requirements = requirements

        verified_ranges = self.__verified_frame_ranges()
        max_window_s = float(self.config.get("coseek1_requirement_candidate_max_window_s") or 24.0)
        unsplit_recovery = bool(
            self.config.get("coseek1_requirement_recovery_unsplit_skim", True)
        )
        recovery_max_window_s = float(
            self.config.get("coseek1_requirement_recovery_max_window_s") or 90.0
        )
        candidates: list[dict[str, Any]] = []

        def verified_overlap_blocks(
            candidate_rng: tuple[float, float],
            *,
            long_unsplit: bool,
        ) -> bool:
            if not long_unsplit:
                return any(
                    self.__range_overlap_ratio(candidate_rng, old_rng) >= 0.35
                    for old_rng in verified_ranges
                )
            candidate_duration = max(1e-6, candidate_rng[1] - candidate_rng[0])
            for old_rng in verified_ranges:
                overlap = min(candidate_rng[1], old_rng[1]) - max(candidate_rng[0], old_rng[0])
                if overlap <= 0:
                    continue
                # A small verified window contained inside a long recovery span
                # should not suppress the whole span; only skip if the long span
                # itself is mostly already verified.
                if overlap / candidate_duration >= 0.85:
                    return True
            return False

        for idx, item in enumerate(self.observation_memory.get("scene_memory") or []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source_tool") or "")
            if source not in {"overview", "skim_qwen", "focus_qwen"}:
                continue
            if source in {"skim_qwen", "focus_qwen"} and not self.__usable_routing_observation(item):
                continue
            windows = [
                self.__valid_range(window)
                for window in (item.get("suggest_focus_windows") or [])
            ]
            windows = [window for window in windows if window is not None]
            item_rng = self.__valid_range(item.get("t_range"))
            if item_rng is not None and self.__scene_memory_item_is_actionable(item):
                windows.append(item_rng)

            for window in windows:
                if window is None:
                    continue
                direct_window = (window[1] - window[0]) <= max_window_s
                window_duration = max(0.0, window[1] - window[0])
                long_unsplit = (
                    unsplit_recovery
                    and not direct_window
                    and window_duration <= recovery_max_window_s
                    and source in {"overview", "skim_qwen", "focus_qwen"}
                )
                if long_unsplit:
                    candidate_ranges = [window]
                else:
                    candidate_ranges = self.__split_range_for_focus_search(
                        window,
                        max_window_s=max_window_s,
                    )
                for candidate_rng in candidate_ranges:
                    if verified_overlap_blocks(candidate_rng, long_unsplit=long_unsplit):
                        continue
                    text, timestamp_hits = self.__candidate_requirement_text(
                        item,
                        candidate_rng,
                    )
                    score = self.__requirement_candidate_score(
                        text,
                        item,
                        source=source,
                        idx=idx,
                        candidate_rng=candidate_rng,
                        requirements=requirements,
                        missing_requirements=missing_requirements,
                            timestamp_hits=timestamp_hits,
                            direct_window=direct_window,
                            long_unsplit=long_unsplit,
                        )
                    if score[0] <= 0:
                        continue
                    candidates.append(
                        {
                            "rng": candidate_rng,
                            "item": item,
                            "source": source,
                            "score": score,
                            "text": text,
                            "long_unsplit": long_unsplit,
                        }
                    )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item["score"], reverse=True)
        deduped: list[dict[str, Any]] = []
        for candidate in candidates:
            rng = candidate["rng"]
            if any(self.__range_overlap_ratio(rng, old["rng"]) >= 0.75 for old in deduped):
                continue
            deduped.append(candidate)
            if len(deduped) >= 4:
                break
        return deduped[0] if deduped else None

    def __candidate_requirement_text(
        self,
        item: dict,
        candidate_rng: tuple[float, float],
    ) -> tuple[str, int]:
        parts = [
            str(item.get("summary") or ""),
            str(item.get("missing_detail") or ""),
            " ".join(str(tag) for tag in item.get("event_tags") or []),
        ]
        timestamp_hits = 0
        for obs in self.observation_memory.get("timestamped_observations") or []:
            if not isinstance(obs, dict):
                continue
            try:
                ts = float(obs.get("timestamp_s"))
            except Exception:
                continue
            if candidate_rng[0] - 2.0 <= ts <= candidate_rng[1] + 2.0:
                timestamp_hits += 1
                parts.append(str(obs.get("description") or ""))
                parts.append(str(obs.get("needs_focus") or ""))
                parts.append(str(obs.get("evidence_detail") or ""))
                parts.extend(str(item) for item in obs.get("matched_requirements") or [])
                parts.extend(str(item) for item in obs.get("missing_requirements") or [])
                parts.extend(str(tag) for tag in obs.get("event_tags") or [])
        return " ".join(part for part in parts if part), timestamp_hits

    def __requirement_candidate_score(
        self,
        text: str,
        item: dict,
        *,
        source: str,
        idx: int,
        candidate_rng: tuple[float, float],
        requirements: list[dict[str, Any]],
        missing_requirements: list[dict[str, Any]],
        timestamp_hits: int,
        direct_window: bool,
        long_unsplit: bool = False,
    ) -> tuple[float, float, float, float, float]:
        text_terms = self.__text_terms(text)

        def matched(req: dict) -> bool:
            terms = set(req.get("terms") or [])
            threshold = int(req.get("threshold") or 1)
            must_any_terms = set(req.get("must_any_terms") or [])
            if must_any_terms and not (must_any_terms & text_terms):
                return False
            return len(terms & text_terms) >= threshold

        missing_hits = sum(1 for req in missing_requirements if matched(req))
        all_hits = sum(1 for req in requirements if matched(req))
        if missing_hits <= 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        source_weight = {"overview": 2.4, "skim_qwen": 2.0, "focus_qwen": 1.6}.get(source, 1.0)
        possible_weight = 0.8 if self.__truthy(item.get("possible_evidence")) else 0.0
        missing_weight = 0.8 if str(item.get("missing_detail") or "").strip() else 0.0
        direct_window_weight = 0.8 if direct_window else 0.0
        long_unsplit_weight = 0.4 if long_unsplit else 0.0
        duration = max(0.0, candidate_rng[1] - candidate_rng[0])
        return (
            5.0 * missing_hits
            + 1.5 * all_hits
            + source_weight
            + possible_weight
            + missing_weight
            + direct_window_weight
            + long_unsplit_weight
            + 0.10 * min(timestamp_hits, 8),
            -abs(duration - 16.0) / 40.0,
            -duration / 200.0,
            -float(idx) * 0.001,
            -candidate_rng[0] / 10000.0,
        )

    def __action_for_requirement_candidate(
        self,
        candidate: dict,
        coverage: dict,
        *,
        step: int,
    ):
        rng = candidate.get("rng")
        if rng is None:
            return None
        item = candidate.get("item") or {}
        missing = coverage.get("critical_missing_requirements") or coverage.get("missing_requirements") or []
        scene_id = str(item.get("scene_id") or "candidate")
        summary = str(item.get("summary") or "").strip()
        query = (
            "Verify the answer using a window that covers all query requirements together.\n"
            f"Question and choices:\n{self.question or ''}\n"
            f"Previous verified support was partial: {coverage.get('support_scope') or ''}\n"
            f"Missing requirements to check in this same window: {missing}\n"
            f"Candidate scene {scene_id} summary: {summary}\n"
            "Do not decide from dress/object color alone unless the target event and "
            "context event are visible in the same evidence window."
        )
        start_time, end_time = rng
        max_window_s = float(self.config.get("coseek1_requirement_candidate_max_window_s") or 24.0)
        if end_time - start_time <= max_window_s + 1e-6:
            return Action(
                function_name="frame_verify",
                parameters={
                    "query": query,
                    "start_time": round(float(start_time), 1),
                    "end_time": round(float(end_time), 1),
                    "mode": "option_verify",
                },
                function_id=f"coseek1_requirement_recovery_frame_verify_{step + 1}",
            )
        return Action(
            function_name="skim_qwen",
            parameters={
                "query": query,
                "start_time": round(float(start_time), 1),
                "end_time": round(float(end_time), 1),
                "mode": "recovery",
                "recovery_skim": True,
            },
            function_id=f"coseek1_requirement_recovery_skim_qwen_{step + 1}",
        )

    def __latest_frame_verify_is_insufficient(self) -> bool:
        for item in reversed(self.observation_memory.get("tool_observations") or []):
            if not isinstance(item, dict):
                continue
            if item.get("tool") != "frame_verify":
                continue
            if item.get("receipt_schema") == "local_visual_receipt_v1":
                # This receipt reports local pixels, not global answer sufficiency.
                return item.get("receipt_parse_ok") is False or item.get("candidate_binding_complete") is False
            supports = item.get("supports_options") or []
            scope = str(item.get("question_scope") or "unknown").strip().lower()
            scope_coverage = str(
                item.get("scope_coverage") or "unknown"
            ).strip().lower()
            if scope == "global_video" and scope_coverage in {
                "partial",
                "insufficient",
            }:
                return True
            if item.get("candidate_binding_complete") is False:
                return True
            detail_sufficient = item.get("detail_sufficient")
            missing_detail = str(item.get("missing_detail") or "").strip()
            coverage = item.get("requirement_coverage")
            if isinstance(coverage, dict):
                critical_missing = (
                    coverage.get("critical_missing_requirements")
                    or coverage.get("missing_requirements")
                    or []
                )
                if coverage.get("support_level") == "partial_verified" and critical_missing:
                    return True
            if detail_sufficient is True and not missing_detail:
                return False
            if detail_sufficient is True and missing_detail:
                lowered = missing_detail.lower()
                if any(
                    token in lowered
                    for token in (
                        "cannot verify",
                        "can't verify",
                        "cannot confirm",
                        "can't confirm",
                        "premise",
                        "not shown",
                        "not visible",
                        "missing required",
                        "missing requirement",
                    )
                ):
                    return True
                return False
            return not bool(supports) and bool(missing_detail)
        return False

    def __select_unverified_overview_candidate(self):
        candidates = [
            item
            for item in self.observation_memory.get("scene_memory") or []
            if isinstance(item, dict)
            and item.get("source_tool") == "overview"
            and self.__truthy(item.get("possible_evidence"))
            and self.__valid_range(item.get("t_range")) is not None
        ]
        if not candidates:
            return None

        verified_ranges = self.__verified_frame_ranges()
        searched_skim_ranges: list[tuple[float, float]] = []
        if self.config.get("coseek1_visited_window_information_gain", False):
            for item in self.observation_memory.get("tool_observations") or []:
                if not isinstance(item, dict) or item.get("tool") != "skim_qwen":
                    continue
                if not self.__usable_routing_observation(item):
                    continue
                params = item.get("parameters") or {}
                rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
                if rng is not None:
                    searched_skim_ranges.append(rng)

        def candidate_is_covered(candidate_range: tuple[float, float]) -> bool:
            candidate_duration = max(1e-6, candidate_range[1] - candidate_range[0])
            short_window_s = float(
                self.config.get("coseek1_requirement_candidate_max_window_s") or 24.0
            )
            for rng in verified_ranges:
                overlap = min(candidate_range[1], rng[1]) - max(candidate_range[0], rng[0])
                if overlap <= 0:
                    continue
                if candidate_duration <= short_window_s:
                    if self.__range_overlap_ratio(candidate_range, rng) >= 0.35:
                        return True
                elif overlap / candidate_duration >= 0.65:
                    return True
            for rng in searched_skim_ranges:
                overlap = min(candidate_range[1], rng[1]) - max(candidate_range[0], rng[0])
                if overlap > 0 and overlap / candidate_duration >= 0.65:
                    return True
            return False

        unverified = []
        for item in candidates:
            candidate_range = self.__valid_range(item.get("t_range"))
            if candidate_range is None:
                continue
            if candidate_is_covered(candidate_range):
                continue
            unverified.append(item)
        if not unverified:
            return None

        def score(item: dict) -> tuple[float, float, float]:
            span = self.__valid_range(item.get("t_range")) or (0.0, self.duration)
            text = f"{item.get('summary') or ''} {item.get('missing_detail') or ''}".lower()
            visual_terms = (
                "hand", "bag", "object", "item", "exchange", "conceal", "take",
                "counter", "checkout", "interaction", "contact", "motion",
                "enter", "leave", "fall", "hit", "fight", "threat", "force",
            )
            term_hits = sum(1 for term in visual_terms if term in text)
            focus_count = len(item.get("suggest_focus_windows") or [])
            missing = 1 if str(item.get("missing_detail") or "").strip() else 0
            # Prefer specific, actionable scenes; break ties by earlier time so
            # skipped evidence before the current window is recovered.
            return (float(term_hits) + 0.5 * focus_count + missing, -span[0], -(span[1] - span[0]))

        return max(unverified, key=score)

    def __verified_frame_ranges(self) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        for item in self.observation_memory.get("tool_observations") or []:
            if not isinstance(item, dict):
                continue
            if item.get("tool") not in {"frame_verify", "focus"}:
                continue
            params = item.get("parameters") or {}
            rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
            if rng is not None:
                ranges.append(rng)
        for item in self.observation_memory.get("scene_memory") or []:
            if not isinstance(item, dict):
                continue
            if item.get("source_tool") not in {"frame_verify", "focus"}:
                continue
            rng = self.__valid_range(item.get("t_range"))
            if rng is not None:
                ranges.append(rng)
        return ranges

    def __action_for_candidate_scene(self, candidate: dict, *, step: int):
        rng = self.__valid_range(candidate.get("t_range"))
        if rng is None:
            return None
        start_time, end_time = rng
        scene_id = str(candidate.get("scene_id") or "candidate")
        summary = str(candidate.get("summary") or "").strip()
        missing = str(candidate.get("missing_detail") or "").strip()
        query = (
            "Continue evidence search before answering.\n"
            f"Question and choices:\n{self.question or ''}\n"
            f"Candidate scene {scene_id} summary: {summary}\n"
            f"Missing detail: {missing}\n"
            "Check whether this scene supports or contradicts any answer option. "
            "If evidence appears, identify the best absolute/global frame_verify window."
        )
        max_focus_window_s = float(self.config.get("focus_qwen_max_window_s") or 20.0)
        suggested_focus = self.__select_memory_suggested_focus_window(
            rng,
            max_window_s=max_focus_window_s,
        )
        if suggested_focus is not None:
            return Action(
                function_name="frame_verify",
                parameters={
                    "query": query,
                    "start_time": round(float(suggested_focus[0]), 1),
                    "end_time": round(float(suggested_focus[1]), 1),
                    "mode": "option_verify",
                },
                function_id=f"coseek1_defer_answer_frame_verify_suggested_{step + 1}",
            )
        if end_time - start_time <= max_focus_window_s:
            return Action(
                function_name="frame_verify",
                parameters={
                    "query": query,
                    "start_time": round(start_time, 1),
                    "end_time": round(end_time, 1),
                    "mode": "option_verify",
                },
                function_id=f"coseek1_defer_answer_frame_verify_{step + 1}",
            )
        return Action(
            function_name="skim_qwen",
            parameters={
                "query": query,
                "start_time": round(start_time, 1),
                "end_time": round(end_time, 1),
                "mode": "normal",
            },
                function_id=f"coseek1_defer_answer_skim_qwen_{step + 1}",
            )

    def __route_frame_verify_through_qwen_focus(
        self,
        actions: List[Action],
        *,
        step: int,
        thought: str,
    ) -> tuple[List[Action], str]:
        actions, thought, context_routed = self.__route_short_verify_with_candidate_context(
            actions,
            step=step,
            thought=thought,
        )
        if context_routed:
            return actions, thought
        if self.config.get("coseek1_intent_preserving_router", False):
            return self.__route_broad_verify_to_same_range_skim(
                actions,
                step=step,
                thought=thought,
            )
        if self.config.get("coseek1_candidate_search_manager", False):
            return self.__route_frame_verify_with_candidate_search(
                actions,
                step=step,
                thought=thought,
            )
        if not self.config.get("coseek1_focus_before_frame_verify", False):
            return actions, thought
        if not self.config.get("coseek1_structured_planner", False):
            return actions, thought
        if not actions or "focus_qwen" not in self.allowed_tool_names:
            return actions, thought

        min_remaining = int(
            self.config.get("coseek1_focus_before_frame_verify_min_remaining_steps") or 2
        )
        remaining_after_this_step = int(self.max_steps) - (step + 1)
        if remaining_after_this_step < min_remaining:
            return actions, thought

        routed_actions: List[Action] = []
        routed_reason = ""
        for idx, action in enumerate(actions, start=1):
            routed, reason = self.__maybe_route_frame_verify_action(
                action,
                step=step,
                index=idx,
            )
            routed_actions.append(routed)
            if reason and not routed_reason:
                routed_reason = reason

        if not routed_reason:
            return actions, thought

        routed_action = routed_actions[0]
        new_thought = (
            "STATE: The requested API frame_verify window has not yet been "
            "localized by focus_qwen, so the router is using local Qwen first.\n"
            f"NEXT_TOOL: {self.__format_action_as_next_tool(routed_action)}\n"
            f"WHY: {routed_reason}"
        )
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages[-1]["content"] = new_thought
        return routed_actions, new_thought

    def __route_short_verify_with_candidate_context(
        self,
        actions: List[Action],
        *,
        step: int,
        thought: str,
    ) -> tuple[List[Action], str, bool]:
        """Preserve nearby candidate boundaries before a short API verify.

        Overview often emits one semantic scene plus several overlapping
        timestamp windows. A planner can reasonably choose one short window,
        but verifying only that slice discards the neighboring transition that
        made the scene a candidate. When that structural ambiguity is present,
        one localize_qwen call inspects the compact union and its existing
        inline child performs the API verification in the same Agent step.

        This does not inspect unrelated scenes, block an answer, or require a
        fixed number of local calls. Single-window candidates remain unchanged.
        """
        if not self.config.get("coseek1_candidate_context_localize_enabled", False):
            return actions, thought, False
        if not actions or "localize_qwen" not in self.allowed_tool_names:
            return actions, thought, False

        routed_actions: List[Action] = []
        routed_reason = ""
        for index, action in enumerate(actions, start=1):
            routed, reason = self.__maybe_localize_candidate_context_action(
                action,
                step=step,
                index=index,
            )
            routed_actions.append(routed)
            if reason and not routed_reason:
                routed_reason = reason

        if not routed_reason:
            return actions, thought, False

        routed_action = routed_actions[0]
        self._candidate_router_reason = routed_reason
        new_thought = self.__format_router_thought(
            routed_action,
            state=(
                "The requested short verification is one slice of a candidate "
                "whose adjacent boundary context has not been inspected."
            ),
            why=routed_reason,
        )
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages[-1]["content"] = new_thought
        return routed_actions, new_thought, True

    def __enrich_localize_with_candidate_context(
        self,
        actions: List[Action],
    ) -> List[Action]:
        """Keep candidate boundary context inside an existing localize call.

        This is cheaper than routing a later verify through an additional
        localize action: the planner still spends exactly one local tool call,
        while Candidate Pool windows prevent a timestamp anchor from becoming
        a hard left/right crop.
        """
        if not self.config.get("coseek1_candidate_context_localize_enabled", False):
            return actions

        enriched: List[Action] = []
        for action in actions:
            if action.function_name != "localize_qwen":
                enriched.append(action)
                continue
            params = dict(action.parameters or {})
            raw_windows = params.get("search_windows") or []
            valid_windows = [
                valid
                for valid in (self.__valid_range(value) for value in raw_windows)
                if valid is not None
            ]
            # Multiple explicit planner windows already express alternative
            # coverage and should not be rewritten by the router.
            if len(valid_windows) != 1:
                enriched.append(action)
                continue
            requested = valid_windows[0]
            focus_limit = float(self.config.get("focus_qwen_max_window_s") or 20.0)
            if requested[1] - requested[0] > focus_limit:
                enriched.append(action)
                continue
            context_windows = self.__candidate_context_windows_for_verify(requested)
            if len(context_windows) < 2:
                enriched.append(action)
                continue
            bounds = (
                min(window[0] for window in context_windows),
                max(window[1] for window in context_windows),
            )
            params["search_windows"] = [list(window) for window in context_windows]
            params["start_time"] = round(bounds[0], 3)
            params["end_time"] = round(bounds[1], 3)
            params["source_planner_search_windows"] = [
                [round(requested[0], 3), round(requested[1], 3)]
            ]
            enriched.append(
                Action(
                    function_name=action.function_name,
                    parameters=params,
                    function_id=action.function_id,
                )
            )
            if not self._candidate_router_reason:
                self._candidate_router_reason = (
                    "enriched the existing localize_qwen call with adjacent "
                    f"Candidate Pool boundary context {self.__format_range(bounds)}; "
                    "no extra tool call was added"
                )
        return enriched

    def __maybe_localize_candidate_context_action(
        self,
        action: Action,
        *,
        step: int,
        index: int,
    ) -> tuple[Action, str]:
        if action.function_name != "frame_verify":
            return action, ""
        params = dict(action.parameters or {})
        requested = self.__valid_range(
            [params.get("start_time"), params.get("end_time")]
        )
        if requested is None:
            return action, ""

        # Broad requests keep the existing same-range skim behavior. This
        # feature only repairs lost context around an already short candidate.
        focus_limit = float(self.config.get("focus_qwen_max_window_s") or 20.0)
        if requested[1] - requested[0] > focus_limit:
            return action, ""

        search_windows = self.__candidate_context_windows_for_verify(requested)
        if len(search_windows) < 2:
            return action, ""
        context_bounds = (
            min(window[0] for window in search_windows),
            max(window[1] for window in search_windows),
        )
        if self.__has_overlapping_tool_window(
            {"localize_qwen"},
            context_bounds,
            threshold=float(
                self.config.get("coseek1_intent_preserving_same_range_overlap")
                or 0.85
            ),
        ):
            return action, ""

        source_id = action.function_id or f"candidate_context_{step + 1}_{index}"
        configured_top_k = int(self.config.get("localize_qwen_top_k") or 3)
        max_top_k = int(self.config.get("localize_qwen_max_top_k") or 4)
        top_k = max(1, min(max_top_k, max(configured_top_k, len(search_windows))))
        mode = str(params.get("mode") or "detail_verify")
        evidence_profile = (
            "event_boundary"
            if mode in {"temporal_strip", "timeline", "count_occurrence"}
            else "generic"
        )
        routed = Action(
            function_name="localize_qwen",
            parameters={
                "search_windows": [list(window) for window in search_windows],
                "localization_goal": str(
                    params.get("query") or self.question or ""
                ).strip(),
                "evidence_profile": evidence_profile,
                "top_k": top_k,
                # Internal routing metadata also makes visited-window auditing
                # work without changing the public tool schema.
                "start_time": round(context_bounds[0], 3),
                "end_time": round(context_bounds[1], 3),
                "source_frame_verify_window": [
                    round(requested[0], 3),
                    round(requested[1], 3),
                ],
            },
            function_id=f"{source_id}_candidate_context_localize",
        )
        reason = (
            "preserved nearby overlapping Candidate Pool windows "
            f"{self.__format_range(context_bounds)} before the same-step API verify"
        )
        return routed, reason

    def __candidate_context_windows_for_verify(
        self,
        requested: tuple[float, float],
    ) -> list[tuple[float, float]]:
        pool = self.observation_memory.get("candidate_pool") or []
        if not isinstance(pool, list):
            return []

        focus_limit = float(self.config.get("focus_qwen_max_window_s") or 20.0)
        radius = float(
            self.config.get("coseek1_persistent_candidate_pool_timestamp_radius_s")
            or 8.0
        )
        adjacency = max(1.0, min(6.0, radius))
        max_span = focus_limit + 2.0 * adjacency
        max_windows = max(
            2,
            min(3, int(self.config.get("localize_qwen_max_search_windows") or 8)),
        )
        min_extension = max(
            1.0,
            0.5 * float(self.config.get("localize_qwen_verify_context_margin_s") or 0.0),
        )

        candidate_options: list[tuple[tuple[float, float, float], list[tuple[float, float]]]] = []
        for item in pool:
            if not isinstance(item, dict) or str(item.get("status") or "") == "verified":
                continue
            windows = [
                valid
                for valid in (
                    self.__valid_range(value)
                    for value in item.get("suggest_focus_windows") or []
                )
                if valid is not None
            ]
            if len(windows) < 2:
                continue
            overlaps = [
                max(0.0, min(requested[1], window[1]) - max(requested[0], window[0]))
                for window in windows
            ]
            best_overlap = max(overlaps or [0.0])
            if best_overlap <= 0.0:
                continue
            try:
                relevance = float(item.get("query_relevance_score") or 0.0)
            except (TypeError, ValueError):
                relevance = 0.0
            candidate_options.append(
                (
                    (
                        best_overlap,
                        relevance,
                        -float((item.get("t_range") or [0.0, self.duration])[0]),
                    ),
                    windows,
                )
            )
        if not candidate_options:
            return []

        _, windows = max(candidate_options, key=lambda item: item[0])
        selected = [requested]
        bounds = requested

        def context_priority(window: tuple[float, float]) -> tuple[float, float, float]:
            overlap = max(
                0.0,
                min(requested[1], window[1]) - max(requested[0], window[0]),
            )
            if overlap > 0.0:
                distance = 0.0
            else:
                distance = min(
                    abs(window[1] - requested[0]),
                    abs(window[0] - requested[1]),
                )
            extension = (
                max(0.0, requested[0] - window[0])
                + max(0.0, window[1] - requested[1])
            )
            return (distance, -overlap, -extension)

        for window in sorted(windows, key=context_priority):
            if any(self.__range_overlap_ratio(window, old) >= 0.98 for old in selected):
                continue
            gap = max(0.0, bounds[0] - window[1], window[0] - bounds[1])
            if gap > adjacency:
                continue
            proposed = (min(bounds[0], window[0]), max(bounds[1], window[1]))
            if proposed[1] - proposed[0] > max_span:
                continue
            selected.append(window)
            bounds = proposed
            if len(selected) >= max_windows:
                break

        extension = (
            max(0.0, requested[0] - bounds[0])
            + max(0.0, bounds[1] - requested[1])
        )
        if len(selected) < 2 or extension < min_extension:
            return []
        return sorted(selected, key=lambda window: (window[0], window[1]))

    def __route_broad_verify_to_same_range_skim(
        self,
        actions: List[Action],
        *,
        step: int,
        thought: str,
    ) -> tuple[List[Action], str]:
        """Preserve planner intent while avoiding an unlocalized broad API call.

        Unlike the V20 candidate-search manager, this router never selects a
        different scene or a heuristic subwindow. It may only replace a broad
        frame_verify with skim_qwen over the exact same parent range and query.
        The resulting captions are exposed in the Candidate Frontier so the
        planner chooses the next short window on the following round.
        """
        if not actions or "skim_qwen" not in self.allowed_tool_names:
            return actions, thought

        max_focus_window_s = float(self.config.get("focus_qwen_max_window_s") or 20.0)
        focus_pad_s = float(self.config.get("focus_qwen_context_pad_s") or 0.0)
        max_short_request_s = max(1.0, max_focus_window_s - 2.0 * focus_pad_s - 0.05)
        overlap_threshold = float(
            self.config.get("coseek1_intent_preserving_same_range_overlap") or 0.85
        )

        routed: List[Action] = []
        for index, action in enumerate(actions, start=1):
            if action.function_name != "frame_verify":
                routed.append(action)
                continue
            params = dict(action.parameters or {})
            rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
            if rng is None or (rng[1] - rng[0]) <= max_short_request_s:
                routed.append(action)
                continue
            already_skimmed = self.__has_overlapping_tool_window(
                {"skim_qwen"},
                rng,
                threshold=overlap_threshold,
            )
            if already_skimmed:
                routed.append(action)
                continue

            source_id = action.function_id or f"v30_planner_verify_{step + 1}_{index}"
            skim_action = Action(
                function_name="skim_qwen",
                parameters={
                    "query": str(params.get("query") or self.question or ""),
                    "start_time": round(float(rng[0]), 1),
                    "end_time": round(float(rng[1]), 1),
                    "mode": "normal",
                },
                function_id=f"{source_id}_same_range_skim",
            )
            routed.append(skim_action)
            if not self._candidate_router_reason:
                self._candidate_router_reason = (
                    "broad frame_verify was converted to skim_qwen over the exact "
                    "planner-requested range; no candidate scene or subwindow was substituted"
                )
        return routed, thought

    def __route_frame_verify_with_candidate_search(
        self,
        actions: List[Action],
        *,
        step: int,
        thought: str,
    ) -> tuple[List[Action], str]:
        if not self.config.get("coseek1_focus_before_frame_verify", False):
            return actions, thought
        if not self.config.get("coseek1_structured_planner", False):
            return actions, thought
        if not actions or "focus_qwen" not in self.allowed_tool_names:
            return actions, thought

        min_remaining = int(
            self.config.get("coseek1_focus_before_frame_verify_min_remaining_steps") or 2
        )
        remaining_after_this_step = int(self.max_steps) - (step + 1)
        if remaining_after_this_step < min_remaining:
            return actions, thought

        routed_actions: List[Action] = []
        routed_reason = ""
        for idx, action in enumerate(actions, start=1):
            routed, reason = self.__maybe_route_frame_verify_candidate_search_action(
                action,
                step=step,
                index=idx,
                thought=thought,
            )
            routed_actions.append(routed)
            if reason and not routed_reason:
                routed_reason = reason

        if not routed_reason:
            return actions, thought

        routed_action = routed_actions[0]
        new_thought = (
            "STATE: Candidate evidence is not localized enough for final API "
            "verification, so local Qwen will inspect another candidate window first.\n"
            f"NEXT_TOOL: {self.__format_action_as_next_tool(routed_action)}\n"
            f"WHY: {routed_reason}"
        )
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages[-1]["content"] = new_thought
        return routed_actions, new_thought

    def __maybe_route_frame_verify_candidate_search_action(
        self,
        action: Action,
        *,
        step: int,
        index: int,
        thought: str,
    ) -> tuple[Action, str]:
        if action.function_name != "frame_verify":
            return action, ""

        params = dict(action.parameters or {})
        rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
        if rng is None:
            return action, ""

        min_window_s = float(
            self.config.get("coseek1_focus_before_frame_verify_min_window_s") or 0.0
        )
        if (rng[1] - rng[0]) <= min_window_s:
            return action, ""

        max_focus_window_s = float(self.config.get("focus_qwen_max_window_s") or 20.0)
        focus_pad_s = float(self.config.get("focus_qwen_context_pad_s") or 0.0)
        max_focus_request_s = max(1.0, max_focus_window_s - 2.0 * focus_pad_s - 0.05)
        overlap_threshold = float(
            self.config.get("coseek1_candidate_search_overlap")
            or self.config.get("coseek1_focus_before_frame_verify_overlap")
            or 0.5
        )
        max_focus = int(
            self.config.get("coseek1_candidate_search_max_focus_before_verify") or 4
        )
        max_candidates = int(self.config.get("coseek1_candidate_search_max_candidates") or 6)

        focus_count = self.__count_overlapping_tool_calls(
            "focus_qwen",
            rng,
            threshold=0.1,
        )
        if focus_count >= max_focus:
            return action, ""

        request_has_focus = self.__has_overlapping_tool_window(
            {"focus_qwen"},
            rng,
            threshold=overlap_threshold,
        )
        request_is_long = (rng[1] - rng[0]) > max_focus_request_s
        if (
            not request_is_long
            and self.config.get(
                "coseek1_candidate_search_allow_short_frame_verify_direct",
                True,
            )
        ):
            return action, ""

        query = str(params.get("query") or self.question or "")
        mode = str(params.get("mode") or "detail_verify")
        source_id = action.function_id or f"coseek1_frame_verify_source_{step + 1}_{index}"
        candidates = self.__candidate_search_windows_for_frame_verify(
            rng,
            max_window_s=max_focus_request_s,
            max_candidates=max_candidates,
        )
        candidate = self.__select_next_unfocused_candidate(
            candidates,
            threshold=overlap_threshold,
        )
        focused_candidate_count = self.__count_focused_candidates(
            candidates,
            threshold=overlap_threshold,
        )
        candidate_count = len(candidates)
        planner_uncertain = self.__planner_text_indicates_uncertain_verify(thought)

        if request_has_focus and not request_is_long:
            return action, ""
        if candidate is None:
            return action, ""

        candidate_from_planner_only = candidate.get("source") == "planner_window"
        should_focus_candidate = False
        route_reason = ""
        if request_is_long and not candidate_from_planner_only:
            should_focus_candidate = True
            route_reason = "the requested API verification span is broad, so Qwen should narrow it to a short candidate first."
        elif focus_count == 0 and candidate_count > 0 and not candidate_from_planner_only:
            should_focus_candidate = True
            route_reason = "there is no usable Qwen localization for this region yet; inspect one short candidate before final API verification."
        elif (
            bool(self.config.get("coseek1_candidate_search_uncertain_text_route", True))
            and planner_uncertain
            and focused_candidate_count == 0
            and not candidate_from_planner_only
        ):
            should_focus_candidate = True
            route_reason = "the planner still describes missing or uncertain evidence, so Qwen should localize another candidate before API verification."
        elif (
            planner_uncertain
            and candidate_count >= 3
            and focus_count < max_focus
            and not candidate_from_planner_only
        ):
            should_focus_candidate = True
            route_reason = "several plausible candidate windows remain and the planner is uncertain; inspect another candidate before spending API."

        if should_focus_candidate:
            candidate_rng = candidate["rng"]
            focus_action = self.__make_focus_qwen_before_verify_action(
                query=query,
                rng=candidate_rng,
                mode=mode,
                function_id=f"{source_id}_candidate_search_focus_qwen",
            )
            return (
                focus_action,
                f"{route_reason} Next candidate: {candidate['source']} {self.__format_range(candidate_rng)}.",
            )

        if (
            self.config.get("coseek1_long_frame_verify_to_skim_qwen", True)
            and "skim_qwen" in self.allowed_tool_names
            and request_is_long
            and not self.__has_overlapping_tool_window(
                {"skim_qwen"},
                rng,
                threshold=overlap_threshold,
            )
        ):
            skim_action = Action(
                function_name="skim_qwen",
                parameters={
                    "query": (
                        "Coarse local candidate search before API frame_verify. "
                        f"{query} Identify multiple plausible short absolute/global "
                        "windows for later focus_qwen."
                    ),
                    "start_time": round(float(rng[0]), 1),
                    "end_time": round(float(rng[1]), 1),
                    "mode": "normal",
                },
                function_id=f"{source_id}_candidate_search_skim_qwen",
            )
            return (
                skim_action,
                "the requested verification span is still broad and no focused candidate pool exists yet.",
            )

        return action, ""

    def __maybe_route_frame_verify_action(
        self,
        action: Action,
        *,
        step: int,
        index: int,
    ) -> tuple[Action, str]:
        if action.function_name != "frame_verify":
            return action, ""

        params = dict(action.parameters or {})
        rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
        if rng is None:
            return action, ""

        min_window_s = float(
            self.config.get("coseek1_focus_before_frame_verify_min_window_s") or 0.0
        )
        if (rng[1] - rng[0]) <= min_window_s:
            return action, ""

        overlap_threshold = float(
            self.config.get("coseek1_focus_before_frame_verify_overlap") or 0.5
        )
        if self.__has_overlapping_tool_window(
            {"focus_qwen"},
            rng,
            threshold=overlap_threshold,
        ):
            return action, ""

        max_focus_window_s = float(self.config.get("focus_qwen_max_window_s") or 20.0)
        focus_pad_s = float(self.config.get("focus_qwen_context_pad_s") or 0.0)
        max_focus_request_s = max(1.0, max_focus_window_s - 2.0 * focus_pad_s - 0.05)
        query = str(params.get("query") or self.question or "")
        mode = str(params.get("mode") or "detail_verify")
        source_id = action.function_id or f"coseek1_frame_verify_source_{step + 1}_{index}"

        if (rng[1] - rng[0]) <= max_focus_request_s:
            focus_action = self.__make_focus_qwen_before_verify_action(
                query=query,
                rng=rng,
                mode=mode,
                function_id=f"{source_id}_focus_qwen_before_frame_verify",
            )
            return (
                focus_action,
                "focus_qwen should first identify the exact frames, then frame_verify can spend API on the narrowed evidence.",
            )

        candidate_window = self.__select_memory_suggested_focus_window(
            rng,
            max_window_s=max_focus_request_s,
        )
        if candidate_window is not None:
            focus_action = self.__make_focus_qwen_before_verify_action(
                query=query,
                rng=candidate_window,
                mode=mode,
                function_id=f"{source_id}_memory_focus_qwen_before_frame_verify",
            )
            return (
                focus_action,
                "a shorter candidate window already exists in memory, so focus_qwen should localize it before API verification.",
            )

        if (
            self.config.get("coseek1_long_frame_verify_to_skim_qwen", True)
            and "skim_qwen" in self.allowed_tool_names
            and not self.__has_overlapping_tool_window(
                {"skim_qwen"},
                rng,
                threshold=overlap_threshold,
            )
        ):
            skim_action = Action(
                function_name="skim_qwen",
                parameters={
                    "query": (
                        "Coarse local routing before API frame_verify. "
                        f"{query} Identify the best short absolute/global window for focus_qwen."
                    ),
                    "start_time": round(float(rng[0]), 1),
                    "end_time": round(float(rng[1]), 1),
                    "mode": "normal",
                },
                function_id=f"{source_id}_skim_qwen_before_frame_verify",
            )
            return (
                skim_action,
                "the requested frame_verify window is still long, so skim_qwen should cheaply narrow it first.",
            )

        clipped = self.__center_crop_range(rng, max_window_s=max_focus_request_s)
        focus_action = self.__make_focus_qwen_before_verify_action(
            query=query,
            rng=clipped,
            mode=mode,
            function_id=f"{source_id}_clipped_focus_qwen_before_frame_verify",
        )
        return (
            focus_action,
            "the requested window is long but has already been skimmed, so focus_qwen inspects a bounded subwindow before API verification.",
        )

    def __make_focus_qwen_before_verify_action(
        self,
        *,
        query: str,
        rng: tuple[float, float],
        mode: str,
        function_id: str,
    ) -> Action:
        return Action(
            function_name="focus_qwen",
            parameters={
                "query": (
                    "Localize exact frames for later frame_verify. "
                    f"{query}"
                ),
                "start_time": round(float(rng[0]), 1),
                "end_time": round(float(rng[1]), 1),
                "mode": "normal" if mode == "detail_verify" else mode,
            },
            function_id=function_id,
        )

    def __candidate_search_windows_for_frame_verify(
        self,
        rng: tuple[float, float],
        *,
        max_window_s: float,
        max_candidates: int,
    ) -> list[dict]:
        candidates: list[dict] = []
        for idx, item in enumerate(self.observation_memory.get("scene_memory") or []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source_tool") or "")
            if source not in {"overview", "skim_qwen", "focus_qwen"}:
                continue
            if source in {"skim_qwen", "focus_qwen"} and not self.__usable_routing_observation(item):
                continue

            item_rng = self.__valid_range(item.get("t_range"))
            windows = [
                self.__valid_range(window)
                for window in (item.get("suggest_focus_windows") or [])
            ]
            windows = [window for window in windows if window is not None]
            if not windows and item_rng is not None and self.__scene_memory_item_is_actionable(item):
                windows = [item_rng]

            for window in windows:
                clipped = self.__clip_range_to_parent(window, rng)
                if clipped is None:
                    continue
                for candidate_rng in self.__split_range_for_focus_search(
                    clipped,
                    max_window_s=max_window_s,
                ):
                    candidates.append(
                        {
                            "rng": candidate_rng,
                            "source": source,
                            "score": self.__candidate_search_score(
                                item,
                                source=source,
                                idx=idx,
                                candidate_rng=candidate_rng,
                            ),
                        }
                    )

        if not candidates:
            for candidate_rng in self.__split_range_for_focus_search(
                rng,
                max_window_s=max_window_s,
            ):
                candidates.append(
                    {
                        "rng": candidate_rng,
                        "source": "planner_window",
                        "score": (0.1, -(candidate_rng[1] - candidate_rng[0]), -candidate_rng[0]),
                    }
                )

        candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
        deduped: list[dict] = []
        for candidate in candidates:
            candidate_rng = candidate["rng"]
            if any(
                self.__range_overlap_ratio(candidate_rng, old["rng"]) >= 0.75
                for old in deduped
            ):
                continue
            deduped.append(candidate)
            if len(deduped) >= max_candidates:
                break
        return deduped

    def __scene_memory_item_is_actionable(self, item: dict) -> bool:
        if self.__truthy(item.get("possible_evidence")):
            return True
        if str(item.get("missing_detail") or "").strip():
            return True
        if item.get("source_tool") in {"skim_qwen", "focus_qwen"}:
            return True
        summary = str(item.get("summary") or "").strip()
        return bool(summary)

    def __candidate_search_score(
        self,
        item: dict,
        *,
        source: str,
        idx: int,
        candidate_rng: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        source_weight = {"skim_qwen": 4.0, "focus_qwen": 3.0, "overview": 2.0}.get(
            source,
            1.0,
        )
        possible_weight = 1.0 if self.__truthy(item.get("possible_evidence")) else 0.0
        missing_weight = 0.4 if str(item.get("missing_detail") or "").strip() else 0.0
        try:
            relevance = float(item.get("relevance") or 0.0)
        except Exception:
            relevance = 0.0
        duration = max(0.0, candidate_rng[1] - candidate_rng[0])
        return (
            source_weight + possible_weight + missing_weight + relevance,
            -duration,
            float(idx) * 0.001,
            -candidate_rng[0],
        )

    def __select_next_unfocused_candidate(
        self,
        candidates: list[dict],
        *,
        threshold: float,
    ) -> dict | None:
        for candidate in candidates:
            rng = candidate.get("rng")
            if rng is None:
                continue
            if self.__has_overlapping_tool_window({"focus_qwen"}, rng, threshold=threshold):
                continue
            return candidate
        return None

    def __count_focused_candidates(
        self,
        candidates: list[dict],
        *,
        threshold: float,
    ) -> int:
        count = 0
        for candidate in candidates:
            rng = candidate.get("rng")
            if rng is None:
                continue
            if self.__has_overlapping_tool_window({"focus_qwen"}, rng, threshold=threshold):
                count += 1
        return count

    def __planner_text_indicates_uncertain_verify(self, thought: str) -> bool:
        if not self.config.get("coseek1_candidate_search_uncertain_text_route", True):
            return False
        text = str(thought or "").lower()
        uncertain_terms = (
            "uncertain",
            "insufficient",
            "not enough",
            "unclear",
            "ambiguous",
            "missing",
            "need",
            "needs",
            "still",
            "not verified",
            "no verified",
            "lack",
            "closer",
            "narrow",
            "candidate",
            "possible",
            "plausible",
            "不确定",
            "不足",
            "缺少",
            "需要",
            "尚未",
            "未验证",
            "候选",
            "可能",
        )
        return any(term in text for term in uncertain_terms)

    def __split_range_for_focus_search(
        self,
        rng: tuple[float, float],
        *,
        max_window_s: float,
    ) -> list[tuple[float, float]]:
        start, end = rng
        if end <= start:
            return []
        if end - start <= max_window_s:
            return [(start, end)]

        stride_ratio = float(
            self.config.get("coseek1_candidate_search_split_stride_ratio") or 0.5
        )
        stride = max(1.0, max_window_s * max(0.25, min(stride_ratio, 1.0)))
        windows: list[tuple[float, float]] = []
        cursor = start
        while cursor < end:
            window = (cursor, min(end, cursor + max_window_s))
            if window[1] > window[0]:
                windows.append(window)
            if window[1] >= end:
                break
            cursor += stride
        return windows

    def __clip_range_to_parent(
        self,
        child: tuple[float, float],
        parent: tuple[float, float],
    ) -> tuple[float, float] | None:
        start = max(float(child[0]), float(parent[0]))
        end = min(float(child[1]), float(parent[1]))
        if end <= start:
            return None
        return start, end

    def __count_overlapping_tool_calls(
        self,
        tool_name: str,
        rng: tuple[float, float],
        *,
        threshold: float,
    ) -> int:
        count = 0
        for item in self.observation_memory.get("tool_observations") or []:
            if not isinstance(item, dict):
                continue
            if item.get("tool") != tool_name:
                continue
            if not self.__usable_routing_observation(item):
                continue
            params = item.get("parameters") or {}
            old_rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
            if old_rng is None:
                continue
            if self.__range_overlap_ratio(rng, old_rng) >= threshold:
                count += 1
        return count

    def __format_range(self, rng: tuple[float, float]) -> str:
        return f"[{float(rng[0]):.1f}, {float(rng[1]):.1f}]"

    def __has_overlapping_tool_window(
        self,
        tool_names: set[str],
        rng: tuple[float, float],
        *,
        threshold: float,
    ) -> bool:
        for item in self.observation_memory.get("tool_observations") or []:
            if not isinstance(item, dict):
                continue
            if item.get("tool") not in tool_names:
                continue
            if not self.__usable_routing_observation(item):
                continue
            params = item.get("parameters") or {}
            old_rng = self.__valid_range([params.get("start_time"), params.get("end_time")])
            if old_rng is not None and self.__range_overlap_ratio(rng, old_rng) >= threshold:
                return True
        for item in self.observation_memory.get("scene_memory") or []:
            if not isinstance(item, dict):
                continue
            if item.get("source_tool") not in tool_names:
                continue
            if not self.__usable_routing_observation(item):
                continue
            old_rng = self.__valid_range(item.get("t_range"))
            if old_rng is not None and self.__range_overlap_ratio(rng, old_rng) >= threshold:
                return True
        return False

    def __usable_routing_observation(self, item: dict) -> bool:
        backend = str(item.get("observer_backend") or "")
        if "not_called" in backend or "error" in backend:
            return False
        if item.get("parsed") is False or item.get("parse_ok") is False:
            return False
        return True

    def __select_memory_suggested_focus_window(
        self,
        rng: tuple[float, float],
        *,
        max_window_s: float,
    ) -> tuple[float, float] | None:
        candidates: list[tuple[tuple[float, float, float, int], tuple[float, float]]] = []
        for idx, item in enumerate(self.observation_memory.get("scene_memory") or []):
            if not isinstance(item, dict):
                continue
            source = item.get("source_tool")
            if source not in {"skim_qwen", "overview", "focus_qwen"}:
                continue
            if source in {"skim_qwen", "focus_qwen"} and not self.__usable_routing_observation(item):
                continue
            source_weight = {"skim_qwen": 3.0, "focus_qwen": 2.0, "overview": 1.0}.get(
                str(source),
                0.0,
            )
            possible_weight = 1.0 if self.__truthy(item.get("possible_evidence")) else 0.0
            for window in item.get("suggest_focus_windows") or []:
                win_rng = self.__valid_range(window)
                if win_rng is None:
                    continue
                overlap = min(rng[1], win_rng[1]) - max(rng[0], win_rng[0])
                if overlap <= 0:
                    continue
                clipped = (max(rng[0], win_rng[0]), min(rng[1], win_rng[1]))
                if clipped[1] - clipped[0] > max_window_s:
                    clipped = self.__center_crop_range(clipped, max_window_s=max_window_s)
                if clipped[1] <= clipped[0]:
                    continue
                if any(
                    self.__range_overlap_ratio(clipped, verified_rng) >= 0.75
                    for verified_rng in self.__verified_frame_ranges()
                ):
                    continue
                score = (
                    source_weight,
                    possible_weight,
                    overlap,
                    idx,
                )
                candidates.append((score, clipped))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def __center_crop_range(
        self,
        rng: tuple[float, float],
        *,
        max_window_s: float,
    ) -> tuple[float, float]:
        start, end = rng
        if end - start <= max_window_s:
            return start, end
        midpoint = (start + end) / 2.0
        half = max_window_s / 2.0
        return max(0.0, midpoint - half), min(self.duration, midpoint + half)

    def __format_action_as_next_tool(self, action: Action) -> str:
        params = action.parameters or {}
        if action.function_name in {"overview", "answer"}:
            return f"{action.function_name}()"
        if action.function_name == "localize_qwen":
            goal = " ".join(
                str(params.get("localization_goal") or "").replace('"', "'").split()
            )
            if len(goal) > 160:
                goal = goal[:157] + "..."
            return (
                f"localize_qwen(search_windows={params.get('search_windows') or []}, "
                f"localization_goal=\"{goal}\", "
                f"evidence_profile=\"{params.get('evidence_profile') or 'generic'}\", "
                f"top_k={int(params.get('top_k') or 1)})"
            )
        query = " ".join(str(params.get("query") or "").replace('"', "'").split())
        if len(query) > 180:
            query = query[:177] + "..."
        return (
            f"{action.function_name}(query=\"{query}\", "
            f"start_time={float(params.get('start_time')):.1f}, "
            f"end_time={float(params.get('end_time')):.1f}, "
            f"mode=\"{params.get('mode') or 'normal'}\")"
        )

    def __format_router_thought(self, action: Action, *, state: str, why: str) -> str:
        if self.config.get("coseek1_planner_json_action", False):
            return json.dumps(
                {
                    "state": state,
                    "action": {
                        "tool": action.function_name,
                        "parameters": action.parameters or {},
                    },
                    "why": why,
                },
                ensure_ascii=True,
            )
        return (
            f"STATE: {state}\n"
            f"NEXT_TOOL: {self.__format_action_as_next_tool(action)}\n"
            f"WHY: {why}"
        )

    def __valid_range(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            start = max(0.0, float(value[0]))
            end = min(float(value[1]), self.duration)
        except Exception:
            return None
        if end <= start:
            return None
        return start, end

    def __range_overlap_ratio(
        self,
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        overlap = min(left[1], right[1]) - max(left[0], right[0])
        if overlap <= 0:
            return 0.0
        denom = max(1e-6, min(left[1] - left[0], right[1] - right[0]))
        return overlap / denom

    def __truthy(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "possible", "likely"}
        return bool(value)

    def __route_action_by_episode_state(
        self,
        actions: List[Action],
        *,
        step: int,
    ) -> List[Action]:
        """Select an observation backend from episode granularity.

        This only fixes a granularity/backend mismatch. It never changes an
        answer action and never requires a fixed number of Qwen calls.
        """
        if not self.config.get("coseek1_episode_backend_router", False) or not actions:
            return actions
        action = actions[0]
        if action.function_name not in {"skim_qwen", "focus_qwen", "frame_verify"}:
            return actions
        params = dict(action.parameters or {})
        start = self.__valid_range([params.get("start_time"), params.get("end_time")])
        if start is None:
            return actions
        start_time, end_time = start
        episode = matching_episode_for_range(
            self.observation_memory,
            [start_time, end_time],
        )
        if not isinstance(episode, dict):
            return actions

        window_s = end_time - start_time
        focus_limit_s = float(
            self.config.get("coseek1_episode_focus_window_s")
            or self.config.get("focus_qwen_max_window_s")
            or 20.0
        )
        broad_limit_s = float(
            self.config.get("coseek1_episode_broad_window_s") or 30.0
        )
        status = str(episode.get("status") or "uninspected")
        routed_tool = action.function_name
        reason = ""

        if action.function_name == "frame_verify" and window_s > broad_limit_s:
            routed_tool = "skim_qwen"
            reason = (
                f"episode {episode.get('episode_id')} is a {window_s:.1f}s broad range; "
                "use one logical adaptive skim before a short semantic verification"
            )
        elif (
            action.function_name == "skim_qwen"
            and window_s <= focus_limit_s
            and status in {"uninspected", "skimmed"}
            and episode.get("origin") in {
                "query_timestamp_anchor", "suggested_window", "caption_event_segment"
            }
        ):
            routed_tool = "focus_qwen"
            reason = (
                f"episode {episode.get('episode_id')} is already a precise "
                f"{window_s:.1f}s anchor"
            )

        if routed_tool == action.function_name:
            return actions
        params["mode"] = "normal" if routed_tool != "frame_verify" else "detail_verify"
        routed = Action(
            function_name=routed_tool,
            parameters=params,
            function_id=f"v31_episode_router_{routed_tool}_{step + 1}",
        )
        self._candidate_router_reason = reason
        return [routed]

    def __p105_overview_completed(self) -> bool:
        return bool(
            self.config.get("coseek1_valid_overview_receipt_guard_enabled", False)
            and has_valid_overview_receipt(self.observation_memory)
        )

    def __planner_action_tools(self) -> list[dict]:
        if not self.__p105_overview_completed():
            return self.tools
        return [
            tool
            for tool in self.tools
            if (tool.get("function") or {}).get("name") != "overview"
        ]

    def __planner_admissible_tool_names(self) -> list[str]:
        return [
            str((tool.get("function") or {}).get("name") or "")
            for tool in self.__planner_action_tools()
            if str((tool.get("function") or {}).get("name") or "")
        ]

    def __memory_capsule(self, full_memory_text: str):
        p100_capsule_enabled = bool(self.config.get("coseek1_planner_decision_neutral_capsule_enabled", False))
        if not (p100_capsule_enabled or self.config.get("coseek1_planner_evidence_capsule_enabled", False)):
            return None
        capsule_builder = (
            build_decision_neutral_planner_capsule
            if p100_capsule_enabled
            else build_planner_evidence_capsule
        )
        capsule_result = capsule_builder(
            self.observation_memory,
            question=self.question,
            full_memory_text=full_memory_text,
            token_budget=int(
                self.config.get("coseek1_planner_capsule_token_budget") or 4000
            ),
            max_verified=int(
                self.config.get("coseek1_planner_capsule_max_verified") or 8
            ),
            max_candidates=int(
                self.config.get("coseek1_planner_capsule_max_candidates") or 8
            ),
            max_obligations=int(
                self.config.get("coseek1_planner_capsule_max_obligations") or 4
            ),
            max_conflicts=2,
        )
        return capsule_result

    def __observe_adaptive_token_budget(self, *, step: int) -> dict[str, Any]:
        status = self._adaptive_token_budget.observe(step_index=step)
        self._adaptive_budget_status = status
        self.observation_memory["adaptive_token_budget"] = dict(status)
        return status

    def __refresh_adaptive_token_budget_total(self) -> dict[str, Any]:
        self._adaptive_token_budget.read_total_tokens()
        status = self._adaptive_token_budget.snapshot()
        self._adaptive_budget_status = status
        self.observation_memory["adaptive_token_budget"] = dict(status)
        return status

    def __allow_remote_api_request(self) -> bool:
        """Gate every remote request while this agent run owns the context."""

        if self._adaptive_final_answer_request_active:
            return True
        status = self.__observe_adaptive_token_budget(
            step=int(getattr(self, "_adaptive_budget_step", 0))
        )
        if status.get("stop"):
            self.observation_memory.setdefault("adaptive_token_budget_skips", []).append(
                {
                    "step": int(getattr(self, "_adaptive_budget_step", 0)) + 1,
                    "tool": "remote_llm_request",
                    "reason": status.get("stop_reason"),
                    "recorded_total_tokens": status.get("recorded_total_tokens"),
                }
            )
            return False
        return True

    def __format_planner_step_prompt(self, *, step: int, memory_text: str) -> str:
        base = (
            f"Step [{step + 1} / {self.max_steps}]: Read the current memory and choose "
            "exactly one next tool.\n\n"
            "Timestamped Observation Memory so far:\n"
            f"{memory_text}\n\n"
        )
        adaptive_budget = getattr(self, "_adaptive_token_budget", None)
        if adaptive_budget is not None:
            base += adaptive_budget.planner_directive(step_index=step)
        overview_completed = self.__p105_overview_completed()
        if overview_completed:
            base += (
                "Executable action state: valid_overview_completed=true; overview is "
                "unavailable; use retained candidate windows.\n\n"
            )
        if self.config.get("coseek1_planner_json_action", False):
            localize_enabled = bool(
                self.config.get("coseek1_localize_qwen_enabled", False)
                and "localize_qwen" in self.allowed_tool_names
            )
            if localize_enabled:
                tool_choices = (
                    "localize_qwen|frame_verify|answer"
                    if overview_completed
                    else "overview|localize_qwen|frame_verify|answer"
                )
            else:
                tool_choices = (
                    "skim_qwen|focus_qwen|frame_verify|answer"
                    if overview_completed
                    else "overview|skim_qwen|focus_qwen|frame_verify|answer"
                )
            if self.config.get("coseek1_evidence_episode_frontier", False):
                investigation_schema = (
                    '"investigation_target":{"episode_id":"EPxxx or null",'
                    '"discriminator":"uncertainty this episode tests",'
                    '"expected_information":"new observation expected"},'
                )
                schema = (
                    '{"state":"short current state",'
                    '"leading_hypothesis":"best current answer hypothesis",'
                    '"strongest_competitor":"strongest alternative hypothesis",'
                    '"decision_critical_evidence":"missing evidence that distinguishes them",'
                    + investigation_schema
                    + f'"action":{{"tool":"{tool_choices}",'
                    '"parameters":{}},"why":"short reason"}'
                )
            elif self.config.get("coseek1_candidate_frontier", False):
                investigation_schema = (
                    '"investigation_target":{"candidate_id":"CFxxx or null",'
                    '"discriminator":"uncertainty this candidate tests",'
                    '"expected_information":"new observation expected"},'
                    if self.config.get("coseek1_frontier_investigation_state", False)
                    else ""
                )
                schema = (
                    '{"state":"short current state",'
                    '"leading_hypothesis":"best current answer hypothesis",'
                    '"strongest_competitor":"strongest alternative hypothesis",'
                    '"decision_critical_evidence":"missing evidence that distinguishes them",'
                    + investigation_schema
                    + f'"action":{{"tool":"{tool_choices}",'
                    '"parameters":{}},"why":"short reason"}'
                )
            else:
                schema = (
                    '{"state":"short current investigation state",'
                    '"leading_hypothesis":"best current answer hypothesis",'
                    '"strongest_competitor":"strongest alternative hypothesis",'
                    '"decision_critical_evidence":"missing evidence that distinguishes them",'
                    '"action":{"tool":'
                    f'"{tool_choices}",'
                    '"parameters":{}},"why":"short reason"}'
                )
            parameter_contract = (
                "\nFor localize_qwen, parameters must include "
                '"search_windows", "localization_goal", "evidence_profile", and "top_k". '
                "Its output already includes coarse and fine local observation; do not split "
                "the same search into separate skim_qwen/focus_qwen actions. "
                if localize_enabled
                else "\nFor skim_qwen and focus_qwen, parameters must include "
                '"query", "start_time", "end_time", and "mode". '
            )
            parameter_contract += (
                "For frame_verify, parameters must include "
                '"query", "start_time", "end_time", and "mode". '
            )
            if self.config.get("coseek1_direct_planner_answer", False):
                parameter_contract += (
                    "For answer, put the selected option and evidence references directly in "
                    'the action as {"tool":"answer","answer":"D",'
                    '"support_refs":["E00128"]}; do not use an empty parameters object. '
                )
            else:
                parameter_contract += "For answer, parameters must be {}. "
            overview_contract = (
                "For overview, parameters must be {}. "
                if not overview_completed
                else ""
            )
            return (
                base
                + "Return exactly one JSON object and no extra text. Use this schema:\n"
                + schema
                + parameter_contract
                + overview_contract
                + "Keep query strings valid JSON so apostrophes such as "
                + "woman's or glasses' remain inside the string."
            )
        tool_choices = (
            "skim_qwen, focus_qwen, frame_verify, or answer"
            if overview_completed
            else "overview, skim_qwen, focus_qwen, frame_verify, or answer"
        )
        return (
            base
            + "Produce exactly the three-line planner output specified by the Planner "
            + "Contract: STATE, NEXT_TOOL, WHY.\n\n"
            + f"Choose exactly one of {tool_choices}. "
            + "No observation is needed in the response."
        )

    def __subtitle_planner_context(self):
        # Enable the shared optional output contract only in the isolated experiment.
        if "subtitle_planner_window_view_enabled" not in self.config or not self.subtitles:
            return convert_to_free_form_text_representation(self.subtitles, content_type="subtitle")
        proposals = []
        for message in self.messages:
            if message.get("role") == "assistant":
                payload = extract_json_object(message.get("content"))
                if isinstance(payload, dict): proposals.append(payload)
        for record in (self.observation_memory.get("p130_global") or {}).get("planner_history", []):
            payload = extract_json_object(record.get("raw_response"))
            if isinstance(payload, dict): proposals.append(payload)
        text, audit = planner_subtitle_view(
            self.subtitles, proposals,
            windowed=bool(self.config["subtitle_planner_window_view_enabled"] and proposals))
        return text + "\n" + SUBTITLE_WINDOW_CONTRACT

    def __subtitle_planner_messages(self, messages):
        if "subtitle_planner_window_view_enabled" not in self.config or not self.subtitles:
            return messages
        full = convert_to_free_form_text_representation(self.subtitles, content_type="subtitle")
        context = self.__subtitle_planner_context()
        # Project actual API inputs only. Preserve the complete original local log.
        return [dict(m, content=m["content"].replace(full, context, 1))
                if isinstance(m.get("content"), str) and full in m["content"] else dict(m)
                for m in messages]

    def __planner_api_messages(self) -> List[dict]:
        messages = self.messages
        if self.config.get("coseek1_compact_planner_context", False) and len(messages) >= 3:
            messages = [dict(messages[0]), dict(messages[1]), dict(messages[-1])]
        return self.__subtitle_planner_messages(messages)

    def __final_answer_api_messages(self) -> List[dict]:
        messages = self.messages
        if self.config.get("coseek1_compact_answer_context", False) and len(messages) >= 3:
            messages = [dict(messages[0]), dict(messages[1]), dict(messages[-1])]
        return self.__subtitle_planner_messages(messages)

    def __p130_execute_and_record(
        self,
        action: Action,
        *,
        thought: str,
        phase: str,
    ) -> str:
        """Execute one global action and append exactly one canonical receipt."""

        step_index = len(self.trajectory_steps)
        self._adaptive_budget_step = self.observation_memory["p130_global"].get("planner_round", step_index)
        if action.function_name != "answer":
            budget = self.__observe_adaptive_token_budget(step=self._adaptive_budget_step)
            if budget.get("stop"):
                return (
                    "Tool execution blocked: adaptive token budget stopped "
                    f"investigation ({budget.get('stop_reason')})."
                )
        self.messages.append(
            {
                "role": "assistant",
                "content": thought,
                "tool_calls": [
                    {
                        "id": action.function_id,
                        "type": "function",
                        "function": {
                            "name": action.function_name,
                            "arguments": str(action.parameters),
                        },
                    }
                ]
                if action.function_name != "answer"
                else [],
            }
        )
        started = time.perf_counter()
        try:
            outcome = self.final_answer if action.function_name == "answer" else self.__exec_action(action)
        except ApiRequestBudgetExceeded as exc:
            outcome = f"Tool execution blocked: {exc}"
        except Exception as exc:
            outcome = f"Tool execution failed: {type(exc).__name__}: {str(exc)[:2000]}"
        elapsed = time.perf_counter() - started
        observation = Observation(action=action, outcome=outcome)

        if action.function_name != "answer":
            append_p130_observation(
                self.observation_memory,
                tool_name=action.function_name,
                parameters=action.parameters,
                output=outcome,
                action_id=str(action.function_id or f"p130_{step_index + 1}"),
                question=str(self.question or ""),
                duration=float(self.duration or 0.0),
            )
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": action.function_id,
                    "content": f"Observation from `{str(action.to_dict())}`:\n{outcome}",
                }
            )

        history = self.observation_memory["p130_global"]["planner_history"]
        proposed = history[-1].get("proposal", {}).get("action") if history else None
        proposed_action = (Action(proposed["tool"], deepcopy(proposed["parameters"]), action.function_id)
                           if proposed else None)
        self.trajectory_steps.append(
            TrajectoryStep(
                step_id=step_index + 1,
                thought=thought,
                action=action,
                observation=observation,
                elapsed_s=elapsed,
                planner_proposed_action=proposed_action,
                routing_audit={
                    "step": step_index + 1,
                    "global_planner": True,
                    "p131_minimal_global_repairs": bool(
                        self.config.get("p131_minimal_global_repairs_enabled", False)
                    ),
                    "phase": phase,
                    "legacy_action_rewrites_bypassed": True,
                },
            )
        )
        self.__refresh_adaptive_token_budget_total()
        if self.verbose:
            label = (
                "P131"
                if self.config.get("p131_minimal_global_repairs_enabled", False)
                else "P130"
            )
            print(f"[{label} {phase}] {action.function_name}: {outcome}")
        return outcome



    def __p130_finish_validated(self, snapshot):
        if not snapshot.get("decision_sufficient"):
            return False
        decision = p130_validate_answer(snapshot.get("validated_option"), snapshot)
        if not decision.get("validated"):
            return False
        self.final_answer = store_global_decision(self.observation_memory, decision)
        self.messages.append({"role": "assistant", "content": self.final_answer})
        return True

    def __p130_finish_forced_prediction(self, snapshot):
        state = self.observation_memory["p130_global"]
        viable = snapshot.get("viable_options") or []
        raw = ""
        claimed = False
        if len(viable) > 1:
            claimed = self._adaptive_token_budget.begin_final_answer()
            if claimed:
                self._adaptive_final_answer_request_active = True
            try:
                response = call_llm_api(
                    messages=[{"role": "user", "content":
                        "Return exactly one option letter from the viable options. This is an "
                        "unvalidated evaluation prediction.\n" + str(self.question) + "\n" +
                        format_p130_snapshot(snapshot) + ("\nVideo Subtitles:\n" + self.__subtitle_planner_context() if self.subtitles else "")}],
                    model_name=self.model_name, api_base=self.api_base, api_key=self.api_key,
                    api_version=self.api_version,
                    max_tokens=min(int(self.max_tokens), int(self._adaptive_token_budget.final_answer_max_tokens)),
                    reasoning_effort=self.reasoning_effort, seed=self.seed, temperature=self.temperature,
                )
                raw = response.choices[0].message.content or ""
            except Exception as exc:
                state["final_request_error"] = type(exc).__name__
            finally:
                self._adaptive_final_answer_request_active = False
                if claimed:
                    self._adaptive_token_budget.complete_final_answer()
        state["forced_prediction_raw"] = raw
        decision = p130_force_answer(raw, snapshot,
                                     reason=state.get("stop_reason", "no_observation_action"))
        self.final_answer = store_global_decision(self.observation_memory, decision)
        self.messages.append({"role": "assistant", "content": self.final_answer})
        self.__refresh_adaptive_token_budget_total()

    def __run_global_planner(self, question: str) -> Trajectory:
        self.reset()
        self.question = question
        init_p130_state(self.observation_memory, question=question, duration=self.duration)
        state = self.observation_memory["p130_global"]
        self.__p130_execute_and_record(
            Action(function_name="overview", parameters={}, function_id="global_overview"),
            thought="Obtain cloud context and independent local routing clues.", phase="overview")
        for index in range(1, self.max_steps):
            state["planner_round"] = self._adaptive_budget_step = index
            budget = self.__observe_adaptive_token_budget(step=index)
            if budget.get("stop"):
                state["stop_reason"] = budget.get("stop_reason") or "token_budget_exhausted"
                break
            messages = [{"role": "system", "content": GLOBAL_PLANNER_INSTRUCTION},
                        {"role": "user", "content": "Question:\n" + question + "\nInvestigation state:\n" +
                         json.dumps(global_planner_state(self.observation_memory, budget=budget), ensure_ascii=False)}]
            if self.subtitles:
                messages[-1]["content"] += "\nVideo Subtitles:\n" + self.__subtitle_planner_context()
            started = time.perf_counter()
            try:
                response = call_llm_api(
                    messages=messages, model_name=self.model_name, api_base=self.api_base,
                    api_key=self.api_key, api_version=self.api_version, max_tokens=min(int(self.max_tokens), 4096),
                    reasoning_effort=self.reasoning_effort, seed=self.seed, temperature=self.temperature,
                    return_json=True)
            except ApiRequestBudgetExceeded:
                state["stop_reason"] = "token_budget_exhausted"
                break
            raw = response.choices[0].message.content or ""
            self.__refresh_adaptive_token_budget_total()
            record = {"round": index, "raw_response": raw, "elapsed_s": time.perf_counter()-started}
            state["planner_history"].append(record)
            try:
                proposal = extract_json_object(raw)
                action_proposal = ({k: v for k, v in proposal.items() if k not in {"subtitle_windows", "subtitle_refs"}}
                                   if "subtitle_planner_window_view_enabled" in self.config and isinstance(proposal, dict) else proposal)
                tool, params = global_planner_action(self.observation_memory, action_proposal)
            except (ValueError, TypeError, KeyError) as exc:
                record["error"] = str(exc)
                if len(state["planner_history"]) > 1 and state["planner_history"][-2].get("error"):
                    state["stop_reason"] = "planner_action_contract_error"
                    break
                continue
            record["proposal"] = proposal
            action = Action(function_name=tool, parameters=params, function_id=f"planner_{index}")
            record["executed_action"] = action.to_dict()
            if tool == "answer":
                state["stop_reason"] = "planner_answer"
                decision = p130_validate_answer(params["answer"], state["snapshot"])
                if not decision.get("validated"):
                    decision = p130_force_answer(params["answer"], state["snapshot"], reason="planner_answer")
                decision["support_refs"] = params["support_refs"]
                self.final_answer = store_global_decision(self.observation_memory, decision)
                self.__p130_execute_and_record(action, thought=proposal["reason"], phase="answer")
                self.messages.append({"role": "assistant", "content": self.final_answer})
                break
            output = self.__p130_execute_and_record(action, thought=proposal["reason"], phase="investigation")
            payload = extract_v10_payload(output) or {}
            if tool == "localize_qwen" and (not payload or payload.get("observer_status") == "unavailable"):
                state["stop_reason"] = "localizer_unavailable"
                break
            if self.__p130_finish_validated(state["snapshot"]):
                break
        else:
            state["stop_reason"] = "planner_step_limit"
        if self.final_answer is None:
            self.__p130_finish_forced_prediction(state["snapshot"])
        terminal = state["terminal_status"]
        reason = {"validated_answer": "p133_validated", "forced_prediction": "p133_forced_prediction",
                  "forced_unresolved": "p133_forced_prediction_unresolved"}[terminal]
        return Trajectory(question=question, steps=self.trajectory_steps, final_answer=self.final_answer,
                          finish_reason=reason, memory=self.observation_memory)

    def run(self, question: str):
        missing_guard = object()
        prior_guard = self.config.get(
            "_adaptive_api_request_guard", missing_guard
        )
        # Observer calls can run in worker threads, where ContextVar state is
        # not inherited.  The config callback closes that gap; the ContextVar
        # still covers planner/action-parser/final calls made directly here.
        request_guard = ApiRequestBudgetGuard(self.__allow_remote_api_request)
        self.config["_adaptive_api_request_guard"] = request_guard
        guard_token = install_api_request_budget_guard(
            request_guard
        )
        try:
            if (
                (
                    self.config.get("p130_minimal_global_fsm_enabled", False)
                    or self.config.get(
                        "p131_minimal_global_repairs_enabled", False
                    )
                )
                and should_use_minimal_global_fsm(question)
            ):
                return self.__run_global_planner(question)
            return self.__run_with_adaptive_budget(question)
        finally:
            reset_api_request_budget_guard(guard_token)
            if prior_guard is missing_guard:
                self.config.pop("_adaptive_api_request_guard", None)
            else:
                self.config["_adaptive_api_request_guard"] = prior_guard

    def __run_with_adaptive_budget(self, question: str):
        self.reset()
        self.question = question
        subtitles_str = convert_to_free_form_text_representation(
            self.subtitles, content_type="subtitle"
        )

        ############################
        # Input
        ############################
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Video Duration: {self.duration:.01f}s\n\n"
                    f"Video Subtitles:\n{subtitles_str}\n\n"
                    f"Question:\n{question}"
                ),
            }
        )

        video_id = self.video_path.split("/")[-1].split(".")[0]
        if self.verbose:
            print("--------------------------------")
            print(f"Video ID: {video_id} ({self.duration:.01f}s)")
            print("--------------------------------")
            print(f"Question:")
            print(question)
            print("--------------------------------")

        for step in range(self.max_steps):
            self._adaptive_budget_step = step
            budget_status = self.__observe_adaptive_token_budget(step=step)
            if budget_status.get("stop"):
                break
            capsule_audit: dict[str, Any] | None = None
            memory_text = format_memory_for_prompt(
                self.observation_memory,
                question=self.question,
                include_compact_planner_state=bool(
                    self.config.get("structured_evidence_include_planner_state", False)
                ),
                include_structured_evidence=bool(
                    self.config.get("use_structured_evidence_state", True)
                ),
                max_structured_evidence_items=int(
                    self.config.get("structured_evidence_prompt_max_items")
                    or self.config.get("structured_evidence_max_items")
                    or 10
                ),
                include_compact_event_coverage=bool(
                    self.config.get("coseek1_compact_event_coverage", False)
                ),
                max_event_coverage_items=int(
                    self.config.get("coseek1_compact_event_coverage_max_items") or 8
                ),
                include_timeline_view=bool(
                    self.config.get("structured_evidence_include_timeline_view", False)
                ),
                include_object_state_view=bool(
                    self.config.get("structured_evidence_include_object_state_view", False)
                ),
                include_candidate_frontier=bool(
                    self.config.get("coseek1_candidate_frontier", False)
                ),
                max_candidate_frontier_items=int(
                    self.config.get("coseek1_candidate_frontier_max_items") or 16
                ),
                candidate_frontier_include_timestamp_cues=bool(
                    self.config.get("coseek1_frontier_timestamp_cues", True)
                ),
                candidate_frontier_timestamp_cue_radius_s=float(
                    self.config.get("coseek1_frontier_timestamp_cue_radius_s") or 6.0
                ),
                candidate_frontier_include_evidence_scope=bool(
                    self.config.get("coseek1_frontier_evidence_scope", True)
                ),
                candidate_frontier_include_temporal_boundaries=bool(
                    self.config.get("coseek1_frontier_temporal_boundaries", True)
                ),
                candidate_frontier_temporal_boundary_max_gap_s=float(
                    self.config.get("coseek1_frontier_temporal_boundary_max_gap_s") or 20.0
                ),
                candidate_frontier_include_investigation_state=bool(
                    self.config.get("coseek1_frontier_investigation_state", False)
                ),
                candidate_frontier_max_investigation_candidates=int(
                    self.config.get("coseek1_frontier_investigation_max_items") or 12
                ),
                include_evidence_episode_frontier=bool(
                    self.config.get("coseek1_evidence_episode_frontier", False)
                ),
                max_evidence_episode_items=int(
                    self.config.get("coseek1_evidence_episode_max_items") or 18
                ),
                evidence_episode_anchor_radius_s=float(
                    self.config.get("coseek1_episode_anchor_radius_s") or 6.0
                ),
                evidence_episode_focus_window_s=float(
                    self.config.get("coseek1_episode_focus_window_s") or 20.0
                ),
                scope_aware_evidence_memory=bool(
                    self.config.get("coseek1_scope_aware_evidence_memory", False)
                ),
                separate_routing_candidates=bool(
                    self.config.get("coseek1_separate_routing_candidates", False)
                ),
                query_relevant_retention=bool(
                    self.config.get("coseek1_query_relevant_memory_retention", False)
                ),
                persistent_candidate_pool=bool(
                    self.config.get("coseek1_persistent_candidate_pool", False)
                ),
                persistent_candidate_pool_max_items=int(
                    self.config.get("coseek1_persistent_candidate_pool_max_items") or 20
                ),
                persistent_candidate_pool_timestamp_radius_s=float(
                    self.config.get("coseek1_persistent_candidate_pool_timestamp_radius_s") or 8.0
                ),
                candidate_evidence_projection=bool(
                    self.config.get("coseek1_candidate_evidence_projection", False)
                ),
                include_temporal_evidence_ledger=bool(
                    self.config.get("coseek1_temporal_evidence_ledger_enabled", False)
                ),
                max_temporal_evidence_events=int(
                    self.config.get("coseek1_temporal_evidence_ledger_max_events") or 16
                ),
            )
            p100_capsule_enabled = bool(
                self.config.get(
                    "coseek1_planner_decision_neutral_capsule_enabled", False
                )
            )
            if (
                self.config.get("coseek1_planner_evidence_capsule_enabled", False)
                or p100_capsule_enabled
            ):
                capsule_result = self.__memory_capsule(memory_text)
                memory_text = capsule_result.text
                if self.config.get("coseek1_planner_capsule_audit_enabled", True):
                    capsule_audit = dict(capsule_result.audit)
                    capsule_audit["step"] = step + 1
                    capsule_audit["capsule_mode"] = (
                        "p100_decision_neutral"
                        if p100_capsule_enabled
                        else "p99_policy_projected"
                    )
                    if self.config.get(
                        "coseek1_valid_overview_receipt_guard_enabled", False
                    ):
                        capsule_audit["p105_admissible_tools"] = (
                            self.__planner_admissible_tool_names()
                        )
                    self.observation_memory.setdefault("planner_capsule_audit", []).append(
                        capsule_audit
                    )
            self.messages.append(
                {
                    "role": "user",
                    "content": self.__format_planner_step_prompt(
                        step=step,
                        memory_text=memory_text,
                    ),
                }
            )

            ############################################################
            # THOUGHT
            ############################################################
            planner_messages = self.__planner_api_messages()
            if capsule_audit is not None:
                planner_text_chars = sum(
                    len(str(message.get("content") or ""))
                    for message in planner_messages
                    if isinstance(message, dict)
                )
                planner_text = "\n".join(
                    str(message.get("content") or "")
                    for message in planner_messages
                    if isinstance(message, dict)
                )
                estimated_prompt_tokens, prompt_tokenizer = estimate_tokens(planner_text)
                capsule_audit["planner_request_message_count"] = len(planner_messages)
                capsule_audit["planner_request_text_chars"] = planner_text_chars
                capsule_audit["estimated_full_planner_prompt_tokens"] = estimated_prompt_tokens
                capsule_audit["planner_prompt_tokenizer"] = prompt_tokenizer
            response = call_llm_api(
                messages=planner_messages,
                model_name=self.model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                seed=self.seed,
                temperature=self.temperature,
            )
            if capsule_audit is not None:
                usage = getattr(response, "usage", None)
                capsule_audit["actual_planner_prompt_tokens"] = int(
                    getattr(usage, "prompt_tokens", 0) or 0
                )
                capsule_audit["actual_planner_completion_tokens"] = int(
                    getattr(usage, "completion_tokens", 0) or 0
                )
                capsule_audit["actual_planner_total_tokens"] = int(
                    getattr(usage, "total_tokens", 0) or 0
                )
            thought = self.__compact_coseek1_thought(response.choices[0].message.content)
            planner_thought = thought
            # thought = """I will use the `overview` tool next to get a 32-frame summary of the entire video, so I can (1) locate where the visit to TeamLab Planets happens in the timeline and (2) see what location appears immediately afterward. This will guide which narrower time segment to inspect with `skim` or `focus` in later steps."""
            self.messages.append({"role": "assistant", "content": thought})
            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] THOUGHT")
                print(thought)
                print("--------------------------------")

            ############################################################
            # ACTIONS
            ############################################################
            planner_payload = extract_json_object(thought)
            if capsule_audit is not None:
                capsule_audit["planner_payload_valid"] = isinstance(planner_payload, dict)
            record_planner_proposal(
                self.observation_memory,
                planner_payload=planner_payload,
                include_investigation_state=bool(
                    self.config.get("coseek1_frontier_investigation_state", False)
                ),
            )
            if self.config.get("coseek1_evidence_episode_frontier", False):
                record_episode_planner_proposal(
                    self.observation_memory,
                    planner_payload=planner_payload,
                )
            actions = self.__parse_actions(thought)
            post_planner_budget = self.__observe_adaptive_token_budget(step=step)
            if post_planner_budget.get("stop") and not (
                actions and all(action.function_name == "answer" for action in actions)
            ):
                break
            planner_proposed_action = None
            if actions:
                proposed = actions[0]
                planner_proposed_action = Action(
                    function_name=proposed.function_name,
                    parameters=dict(proposed.parameters or {}),
                    function_id=proposed.function_id,
                )
            self._candidate_router_reason = ""
            actions = self.__force_initial_observation_if_needed(actions, step=step)
            actions = self.__auto_probe_scene_before_answer(actions, step=step)
            actions = self.__auto_prepend_scene_skim_before_focus(actions, step=step)
            actions = self.__qwen_first_route_actions(actions, step=step)
            actions = self.__route_action_by_episode_state(actions, step=step)
            actions, thought = self.__defer_answer_if_evidence_insufficient(
                actions,
                step=step,
                thought=thought,
            )
            actions = self.__enrich_localize_with_candidate_context(actions)
            actions, thought = self.__route_frame_verify_through_qwen_focus(
                actions,
                step=step,
                thought=thought,
            )
            actions, thought = self.__route_repeated_skim_for_information_gain(
                actions,
                step=step,
                thought=thought,
            )
            if self.config.get("coseek1_preserve_planner_thought", False):
                thought = planner_thought
                if self.messages and self.messages[-1].get("role") == "assistant":
                    self.messages[-1]["content"] = planner_thought
            executed_action = actions[0] if actions else None
            if capsule_audit is not None:
                proposed_dict = (
                    planner_proposed_action.to_dict() if planner_proposed_action else None
                )
                executed_dict = executed_action.to_dict() if executed_action else None
                capsule_audit["planner_proposed_action"] = proposed_dict
                capsule_audit["executed_action"] = executed_dict
                action_parameters = (
                    dict(executed_action.parameters or {}) if executed_action else {}
                )
                investigation = (
                    planner_payload.get("investigation_target")
                    if isinstance(planner_payload, dict)
                    and isinstance(planner_payload.get("investigation_target"), dict)
                    else {}
                )
                selected_candidate_id = str(
                    investigation.get("candidate_id")
                    or action_parameters.get("candidate_id")
                    or ""
                )
                capsule_audit["selected_candidate_id"] = selected_candidate_id or None
                capsule_audit["selected_event_id"] = (
                    (capsule_audit.get("candidate_event_map") or {}).get(
                        selected_candidate_id
                    )
                    if selected_candidate_id
                    else None
                )
                capsule_audit["support_refs"] = list(
                    action_parameters.get("support_refs") or []
                )
                logical_identity = capsule_audit.get("selected_event_id")
                if (
                    self.config.get(
                        "coseek1_valid_overview_receipt_guard_enabled", False
                    )
                    and executed_action is not None
                ):
                    signature_parameters = dict(action_parameters)
                    if selected_candidate_id:
                        signature_parameters.setdefault(
                            "candidate_id", selected_candidate_id
                        )
                    if capsule_audit.get("selected_event_id"):
                        signature_parameters.setdefault(
                            "event_id", capsule_audit["selected_event_id"]
                        )
                    logical_identity = canonical_action_signature(
                        executed_action.function_name,
                        signature_parameters,
                    )
                elif not logical_identity and executed_action is not None:
                    logical_identity = (
                        f"{executed_action.function_name}:"
                        f"{action_parameters.get('start_time')}:"
                        f"{action_parameters.get('end_time')}"
                    )
                capsule_audit["selected_logical_identity"] = logical_identity
                prior_audits = self.observation_memory.get("planner_capsule_audit") or []
                prior_identity = (
                    prior_audits[-2].get("selected_logical_identity")
                    if len(prior_audits) >= 2
                    else None
                )
                capsule_audit["repeats_previous_logical_event"] = bool(
                    logical_identity and prior_identity == logical_identity
                )
            routing_audit = record_routing_audit(
                self.observation_memory,
                step=step + 1,
                planner_action=planner_proposed_action,
                executed_action=executed_action,
                reason=self._candidate_router_reason,
            )
            overview_guard_will_block = bool(
                executed_action is not None
                and executed_action.function_name == "overview"
                and self.__p105_overview_completed()
            )
            if overview_guard_will_block:
                routing_audit["executor_blocked"] = True
                routing_audit["executor_guard"] = (
                    "valid_overview_already_completed"
                )
            if self.verbose:
                print(f"[STEP {step+1} / {self.max_steps}] ACTIONS")
                print([str(action) for action in actions])
                print("--------------------------------")
            if len(actions) != 0 and actions[0].function_name != "answer":
                self.messages[-1]["tool_calls"] = [
                    {
                        "id": action.function_id,
                        "type": "function",
                        "function": {
                            "name": action.function_name,
                            "arguments": str(action.parameters),
                        },
                    }
                    for action in actions
                ]

            ############################################################
            # OBSERVATIONS
            ############################################################
            for action in actions:
                dispatch_budget = self.__observe_adaptive_token_budget(step=step)
                if dispatch_budget.get("stop") and action.function_name != "answer":
                    self.observation_memory.setdefault(
                        "adaptive_token_budget_skips", []
                    ).append(
                        {
                            "step": step + 1,
                            "tool": action.function_name,
                            "reason": dispatch_budget.get("stop_reason"),
                            "recorded_total_tokens": dispatch_budget.get(
                                "recorded_total_tokens"
                            ),
                        }
                    )
                    break
                p105_blocked_overview = bool(
                    action.function_name == "overview"
                    and self.__p105_overview_completed()
                )
                action_t0 = time.perf_counter()
                adaptive_final_claimed = False
                if (
                    action.function_name == "answer"
                    and dispatch_budget.get("stop")
                    and self._adaptive_token_budget.enabled
                ):
                    adaptive_final_claimed = (
                        self._adaptive_token_budget.begin_final_answer()
                    )
                    if not adaptive_final_claimed:
                        break
                    self._adaptive_final_answer_request_active = True
                try:
                    outcome = self.__exec_action(action)
                except ApiRequestBudgetExceeded as e:
                    outcome = f"Tool execution blocked: {e}"
                except Exception as e:
                    outcome = (
                        "Tool execution failed: "
                        f"{type(e).__name__}: {str(e)[:2000]}"
                    )
                finally:
                    if adaptive_final_claimed:
                        self._adaptive_final_answer_request_active = False
                        self._adaptive_token_budget.complete_final_answer()
                        self.__refresh_adaptive_token_budget_total()
                action_elapsed_s = time.perf_counter() - action_t0
                if self.verbose:
                    print(f"[STEP {step+1} / {self.max_steps}] OBSERVATION")
                    print(outcome)
                    print("--------------------------------")
                observation = Observation(action=action, outcome=outcome)
                if action.function_name != "answer" and not p105_blocked_overview:
                    merge_tool_observation(
                        self.observation_memory,
                        tool_name=action.function_name,
                        parameters=action.parameters,
                        output=outcome,
                        use_structured_evidence_state=bool(
                            self.config.get("use_structured_evidence_state", True)
                        ),
                        question_context=self.question,
                        include_evidence_scope=bool(
                            self.config.get("coseek1_frontier_evidence_scope", True)
                        ),
                        scope_aware_evidence_memory=bool(
                            self.config.get("coseek1_scope_aware_evidence_memory", False)
                        ),
                        persistent_open_gaps=bool(
                            self.config.get("coseek1_persistent_open_gaps", False)
                        ),
                        semantic_gap_resolution=bool(
                            self.config.get(
                                "coseek1_semantic_gap_resolution_enabled", False
                            )
                        ),
                        preserve_aggregate_verifier_decision=bool(
                            self.config.get(
                                "structured_evidence_preserve_aggregate_verifier_decision",
                                False,
                            )
                        ),
                    )
                    if (
                        action.function_name == "overview"
                        and self.config.get(
                            "coseek1_valid_overview_receipt_guard_enabled", False
                        )
                    ):
                        record_overview_attempt(
                            self.observation_memory,
                            output=outcome,
                            require_all_timestamps=bool(
                                self.config.get("overview_require_all_timestamps", False)
                            ),
                            step=step + 1,
                        )
                    self.__attach_last_tool_observation_timing(
                        tool_name=action.function_name,
                        elapsed_s=action_elapsed_s,
                    )
                    self.__annotate_last_requirement_coverage(
                        tool_name=action.function_name,
                        parameters=action.parameters,
                        output=outcome,
                    )
                    if self.config.get("coseek1_candidate_frontier", False):
                        refresh_candidate_frontier(
                            self.observation_memory,
                            question=self.question,
                            duration=self.duration,
                            max_candidates=int(
                                self.config.get("coseek1_candidate_frontier_max_items") or 16
                            )
                            + 8,
                            include_timestamp_cues=bool(
                                self.config.get("coseek1_frontier_timestamp_cues", True)
                            ),
                            timestamp_cue_radius_s=float(
                                self.config.get("coseek1_frontier_timestamp_cue_radius_s") or 6.0
                            ),
                            include_evidence_scope=bool(
                                self.config.get("coseek1_frontier_evidence_scope", True)
                            ),
                            include_temporal_boundaries=bool(
                                self.config.get("coseek1_frontier_temporal_boundaries", True)
                            ),
                            temporal_boundary_max_gap_s=float(
                                self.config.get("coseek1_frontier_temporal_boundary_max_gap_s") or 20.0
                            ),
                        )
                    if self.config.get("coseek1_evidence_episode_frontier", False):
                        refresh_evidence_episode_frontier(
                            self.observation_memory,
                            question=self.question,
                            duration=self.duration,
                            max_episodes=int(
                                self.config.get("coseek1_evidence_episode_max_items") or 18
                            )
                            + 48,
                            anchor_radius_s=float(
                                self.config.get("coseek1_episode_anchor_radius_s") or 6.0
                            ),
                            focus_window_s=float(
                                self.config.get("coseek1_episode_focus_window_s") or 20.0
                            ),
                        )
                if action.parameters is not None:
                    action.parameters.pop("vr", None)
                    action.parameters.pop("subtitles", None)
                self.trajectory_steps.append(
                    TrajectoryStep(
                        step_id=step + 1,
                        thought=thought,
                        action=action,
                        observation=observation,
                        elapsed_s=action_elapsed_s,
                        planner_proposed_action=planner_proposed_action,
                        routing_audit=routing_audit,
                    )
                )
                if action.function_name == "answer":
                    self.final_answer = observation.outcome
                    if self.config.get("coseek1_answer_evidence_audit", False):
                        record_answer_evidence_audit(
                            self.observation_memory,
                            answer=self.final_answer,
                            step=step + 1,
                            planner_payload=planner_payload,
                        )
                    break
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": action.function_id,
                        "content": f"Observation from `{str(action.to_dict())}`:\n{outcome}",
                    }
                )
                inline_verify_action = self.__inline_localize_verify_action(
                    localize_action=action,
                    localize_output=outcome,
                    step=step,
                )
                if inline_verify_action is not None:
                    inline_budget = self.__observe_adaptive_token_budget(step=step)
                    if inline_budget.get("stop"):
                        self.observation_memory.setdefault(
                            "adaptive_token_budget_skips", []
                        ).append(
                            {
                                "step": step + 1,
                                "tool": inline_verify_action.function_name,
                                "reason": inline_budget.get("stop_reason"),
                                "recorded_total_tokens": inline_budget.get(
                                    "recorded_total_tokens"
                                ),
                            }
                        )
                        inline_verify_action = None
                if inline_verify_action is not None:
                    self.__append_tool_call_message(inline_verify_action)
                    inline_t0 = time.perf_counter()
                    try:
                        inline_outcome = self.__exec_action(inline_verify_action)
                    except Exception as e:
                        inline_outcome = (
                            "Tool execution failed: "
                            f"{type(e).__name__}: {str(e)[:2000]}"
                        )
                    inline_elapsed_s = time.perf_counter() - inline_t0
                    if self.verbose:
                        print(f"[STEP {step+1} / {self.max_steps}] INLINE MULTI-WINDOW VERIFY")
                        print(inline_outcome)
                        print("--------------------------------")
                    inline_observation = Observation(
                        action=inline_verify_action,
                        outcome=inline_outcome,
                    )
                    merge_tool_observation(
                        self.observation_memory,
                        tool_name=inline_verify_action.function_name,
                        parameters=inline_verify_action.parameters,
                        output=inline_outcome,
                        use_structured_evidence_state=bool(
                            self.config.get("use_structured_evidence_state", True)
                        ),
                        question_context=self.question,
                        include_evidence_scope=bool(
                            self.config.get("coseek1_frontier_evidence_scope", True)
                        ),
                        scope_aware_evidence_memory=bool(
                            self.config.get("coseek1_scope_aware_evidence_memory", False)
                        ),
                        persistent_open_gaps=bool(
                            self.config.get("coseek1_persistent_open_gaps", False)
                        ),
                        semantic_gap_resolution=bool(
                            self.config.get(
                                "coseek1_semantic_gap_resolution_enabled", False
                            )
                        ),
                        preserve_aggregate_verifier_decision=bool(
                            self.config.get(
                                "structured_evidence_preserve_aggregate_verifier_decision",
                                False,
                            )
                        ),
                    )
                    self.__attach_last_tool_observation_timing(
                        tool_name=inline_verify_action.function_name,
                        elapsed_s=inline_elapsed_s,
                    )
                    self.__annotate_last_requirement_coverage(
                        tool_name=inline_verify_action.function_name,
                        parameters=inline_verify_action.parameters,
                        output=inline_outcome,
                    )
                    self.trajectory_steps.append(
                        TrajectoryStep(
                            step_id=step + 1,
                            thought=(
                                thought
                                + "\n\n[V46 tool continuation] localize_qwen produced usable "
                                "candidates; one API frame_verify compared them in the same "
                                "logical investigation step."
                            ),
                            action=inline_verify_action,
                            observation=inline_observation,
                            elapsed_s=inline_elapsed_s,
                            planner_proposed_action=planner_proposed_action,
                            routing_audit={
                                "step": step + 1,
                                "router_changed_action": False,
                                "router_reason": "inline_child_of_localize_qwen",
                                "parent_action_id": action.function_id,
                            },
                        )
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": inline_verify_action.function_id,
                            "content": (
                                f"Observation from `{str(inline_verify_action.to_dict())}`:\n"
                                f"{inline_outcome}"
                            ),
                        }
                    )
                if self.__qwen_first_should_verify_with_api(action, outcome):
                    fallback_action = self._qwen_first_api_fallbacks.get(action.function_id)
                    if fallback_action is not None:
                        fallback_budget = self.__observe_adaptive_token_budget(step=step)
                        if fallback_budget.get("stop"):
                            self.observation_memory.setdefault(
                                "adaptive_token_budget_skips", []
                            ).append(
                                {
                                    "step": step + 1,
                                    "tool": fallback_action.function_name,
                                    "reason": fallback_budget.get("stop_reason"),
                                    "recorded_total_tokens": fallback_budget.get(
                                        "recorded_total_tokens"
                                    ),
                                }
                            )
                            fallback_action = None
                    if fallback_action is not None:
                        self.__append_tool_call_message(fallback_action)
                        fallback_t0 = time.perf_counter()
                        try:
                            fallback_outcome = self.__exec_action(fallback_action)
                        except Exception as e:
                            fallback_outcome = (
                                "Tool execution failed: "
                                f"{type(e).__name__}: {str(e)[:2000]}"
                            )
                        fallback_elapsed_s = time.perf_counter() - fallback_t0
                        if self.verbose:
                            print(f"[STEP {step+1} / {self.max_steps}] QWEN-FIRST API VERIFY")
                            print(fallback_outcome)
                            print("--------------------------------")
                        fallback_observation = Observation(
                            action=fallback_action,
                            outcome=fallback_outcome,
                        )
                        merge_tool_observation(
                            self.observation_memory,
                            tool_name=fallback_action.function_name,
                            parameters=fallback_action.parameters,
                            output=fallback_outcome,
                            use_structured_evidence_state=bool(
                                self.config.get("use_structured_evidence_state", True)
                            ),
                            question_context=self.question,
                            include_evidence_scope=bool(
                                self.config.get("coseek1_frontier_evidence_scope", True)
                            ),
                            scope_aware_evidence_memory=bool(
                                self.config.get("coseek1_scope_aware_evidence_memory", False)
                            ),
                            persistent_open_gaps=bool(
                                self.config.get("coseek1_persistent_open_gaps", False)
                            ),
                            semantic_gap_resolution=bool(
                                self.config.get(
                                    "coseek1_semantic_gap_resolution_enabled", False
                                )
                            ),
                            preserve_aggregate_verifier_decision=bool(
                                self.config.get(
                                    "structured_evidence_preserve_aggregate_verifier_decision",
                                    False,
                                )
                            ),
                        )
                        self.__attach_last_tool_observation_timing(
                            tool_name=fallback_action.function_name,
                            elapsed_s=fallback_elapsed_s,
                        )
                        self.__annotate_last_requirement_coverage(
                            tool_name=fallback_action.function_name,
                            parameters=fallback_action.parameters,
                            output=fallback_outcome,
                        )
                        self.trajectory_steps.append(
                            TrajectoryStep(
                                step_id=step + 1,
                                thought=(
                                    thought
                                    + "\n\n[Qwen-first router] Local Qwen observation was insufficient; "
                                    "ran API verification on the same logical action."
                                ),
                                action=fallback_action,
                                observation=fallback_observation,
                                elapsed_s=fallback_elapsed_s,
                            )
                        )
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": fallback_action.function_id,
                                "content": (
                                    f"Observation from `{str(fallback_action.to_dict())}`:\n"
                                    f"{fallback_outcome}"
                                ),
                            }
                        )
            
            if len(actions) == 0:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "There is no function call in your response. YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.",
                    }
                )
                continue

            ############################################################
            # STOP IF FINAL ANSWER IS FOUND
            ############################################################
            if self.final_answer is not None:
                break

        ############################################################
        # REACH MAX STEPS BUT NO FINAL ANSWER IS FOUND
        ############################################################
        if self.final_answer is None:
            adaptive_stop_reason = self._adaptive_token_budget.stop_reason
            evidence_text = ""
            if self.config.get("use_evidence_reducer", True):
                evidence_text = format_evidence_for_answer(
                    self.observation_memory,
                    question=self.question,
                    include_compact_planner_state=bool(
                        self.config.get("structured_evidence_include_planner_state", False)
                    ),
                    include_structured_evidence=bool(
                        self.config.get("use_structured_evidence_state", True)
                    ),
                    max_structured_evidence_items=int(
                        self.config.get("structured_evidence_answer_max_items")
                        or self.config.get("structured_evidence_max_items")
                        or 16
                    ),
                    include_timeline_view=bool(
                        self.config.get("structured_evidence_include_timeline_view", False)
                    ),
                    include_object_state_view=bool(
                        self.config.get("structured_evidence_include_object_state_view", False)
                    ),
                    scope_aware_evidence_memory=bool(
                        self.config.get("coseek1_scope_aware_evidence_memory", False)
                    ),
                    separate_routing_candidates=bool(
                        self.config.get("coseek1_separate_routing_candidates", False)
                    ),
                    query_relevant_retention=bool(
                        self.config.get("coseek1_query_relevant_memory_retention", False)
                    ),
                    include_temporal_evidence_ledger=bool(
                        self.config.get("coseek1_temporal_evidence_ledger_enabled", False)
                    ),
                    max_temporal_evidence_events=int(
                        self.config.get("coseek1_temporal_evidence_ledger_max_events") or 16
                    ),
                )
            capsule_result = self.__memory_capsule(evidence_text)
            if capsule_result is not None:
                evidence_text = capsule_result.text
                if self.config.get("coseek1_planner_capsule_audit_enabled", True):
                    self.observation_memory.setdefault("planner_capsule_audit", []).append(
                        dict(capsule_result.audit, phase="final_answer"))
            limit_message = (
                "You have reached the adaptive investigation token budget. "
                "Do not request more evidence. "
                if adaptive_stop_reason
                else "You have reached the maximum number of steps. "
            )
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        limit_message
                        +
                        "Use the evidence digest below to compare all collected evidence before answering.\n\n"
                        f"Evidence digest:\n{evidence_text}\n\n"
                        f"Question:\n{question}\n\n"
                        "If the question is a multiple-choice question, please directly answer with the option's letter from the given choices without any additional text. "
                        "Use the option-wise evidence table first when it is present; verified support is stronger than weak_support, and weak_support is not decisive when detail_sufficient=false."
                    ),
                }
            )
            adaptive_final_claimed = False
            if adaptive_stop_reason:
                adaptive_final_claimed = self._adaptive_token_budget.begin_final_answer()
                if not adaptive_final_claimed:
                    raise RuntimeError(
                        "compact final-answer request was already consumed"
                    )
                self._adaptive_final_answer_request_active = True
            try:
                response = call_llm_api(
                    messages=self.__final_answer_api_messages(),
                    model_name=self.model_name,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    max_tokens=(
                        min(
                            self.max_tokens,
                            self._adaptive_token_budget.final_answer_max_tokens,
                        )
                        if adaptive_stop_reason
                        else self.max_tokens
                    ),
                    reasoning_effort=self.reasoning_effort,
                    seed=self.seed,
                    temperature=self.temperature,
                )
            finally:
                if adaptive_final_claimed:
                    self._adaptive_final_answer_request_active = False
                    self._adaptive_token_budget.complete_final_answer()
            self.final_answer = response.choices[0].message.content
            self.__refresh_adaptive_token_budget_total()
            if self.config.get("coseek1_answer_evidence_audit", False):
                record_answer_evidence_audit(
                    self.observation_memory,
                    answer=self.final_answer,
                    step=self.max_steps,
                    planner_payload=None,
                )
            self.messages.append({"role": "assistant", "content": self.final_answer})
            return Trajectory(
                question=question,
                steps=self.trajectory_steps,
                final_answer=self.final_answer,
                finish_reason=(
                    self._adaptive_token_budget.finish_reason()
                    if adaptive_stop_reason
                    else "reach_max_steps"
                ),
                memory=self.observation_memory,
            )

        self.__refresh_adaptive_token_budget_total()
        adaptive_finish_reason = self._adaptive_token_budget.finish_reason()
        return Trajectory(
            question=question,
            steps=self.trajectory_steps,
            final_answer=self.final_answer,
            finish_reason=adaptive_finish_reason or "stop",
            memory=self.observation_memory,
        )
