from typing import List, Dict, Any
from .action import Action
from .observation import Observation


class TrajectoryStep:
    """A single step in the agent trajectory."""

    def __init__(
        self,
        step_id: int,
        thought: str,
        action: Action,
        observation: Observation,
        elapsed_s: float | None = None,
        planner_proposed_action: Action | None = None,
        routing_audit: Dict[str, Any] | None = None,
    ):
        self.step_id = step_id
        self.thought = thought
        self.action = action
        self.observation = observation
        self.elapsed_s = elapsed_s
        self.planner_proposed_action = planner_proposed_action
        self.routing_audit = routing_audit

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "step_id": self.step_id,
            "thought": self.thought,
            "action": self.action.to_dict(),
            "observation": str(self.observation),
        }
        if self.elapsed_s is not None:
            data["elapsed_s"] = round(float(self.elapsed_s), 3)
        if self.planner_proposed_action is not None:
            data["planner_proposed_action"] = self.planner_proposed_action.to_dict()
        if self.routing_audit is not None:
            data["routing_audit"] = self.routing_audit
        return data


class Trajectory:
    """Complete trajectory of an agent run."""
    def __init__(
        self,
        question: str,
        steps: List[TrajectoryStep],
        final_answer: str,
        finish_reason: str,
        memory: Dict[str, Any] | None = None,
    ):
        self.question = question
        self.steps = steps
        self.final_answer = final_answer
        self.finish_reason = finish_reason
        self.memory = memory or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "total_steps": max((s.step_id for s in self.steps), default=0),
            "final_answer": self.final_answer,
            "finish_reason": self.finish_reason,
            "memory": self.memory,
        }
