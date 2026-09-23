from videoseek.utils import call_llm_api
from videoseek.core.memory import format_evidence_for_answer


answer_tool = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Based on the given trajectory, generate the final answer to the question.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def execute_answer(config: dict, parameters: dict) -> str:
    """
    Execute the answer tool.
    """
    question = parameters['question']
    messages = parameters['messages']
    memory = parameters.get("memory")
    evidence_text = ""
    if config.get("use_evidence_reducer", True):
        evidence_text = format_evidence_for_answer(
            memory,
            question=question,
            include_compact_planner_state=bool(
                config.get("structured_evidence_include_planner_state", False)
            ),
            include_structured_evidence=bool(
                config.get("use_structured_evidence_state", True)
            ),
            max_structured_evidence_items=int(
                config.get("structured_evidence_answer_max_items")
                or config.get("structured_evidence_max_items")
                or 16
            ),
            include_timeline_view=bool(
                config.get("structured_evidence_include_timeline_view", False)
            ),
            include_object_state_view=bool(
                config.get("structured_evidence_include_object_state_view", False)
            ),
            scope_aware_evidence_memory=bool(
                config.get("coseek1_scope_aware_evidence_memory", False)
            ),
            separate_routing_candidates=bool(
                config.get("coseek1_separate_routing_candidates", False)
            ),
            query_relevant_retention=bool(
                config.get("coseek1_query_relevant_memory_retention", False)
            ),
            include_temporal_evidence_ledger=bool(
                config.get("coseek1_temporal_evidence_ledger_enabled", False)
            ),
            max_temporal_evidence_events=int(
                config.get("coseek1_temporal_evidence_ledger_max_events") or 16
            ),
        )
    evidence_prefix = (
        "Evidence digest before final answer:\n"
        f"{evidence_text}\n\n"
        if evidence_text
        else ""
    )
    messages.append({
        "role": "user",
        "content": (
            evidence_prefix
            +
            f"Question:\n{question}\n\n"
            "Please directly provide the final answer. For multiple-choice questions, output only the option letter.\n"
            "Use the option-wise evidence table first when it is present. Verified support is stronger than weak_support; "
            "weak_support is not decisive if the same option is contradicted or detail_sufficient=false. "
            "Do not choose a contradicted option just because related free-text evidence mentions its words."
        ),
    })
    response = call_llm_api(
        messages=messages,
        model_name=config['model_name'],
        api_base=config['api_base'],
        api_key=config['api_key'],
        api_version=config['api_version'],
        max_tokens=config['max_tokens'],
        reasoning_effort=config['reasoning_effort'],
        seed=config['seed'],
        temperature=config['temperature'])

    return response.choices[0].message.content
