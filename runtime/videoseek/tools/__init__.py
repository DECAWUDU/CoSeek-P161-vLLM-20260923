from .overview import overview_tool, execute_overview
from .skim import skim_tool, execute_skim
from .skim_qwen import skim_qwen_tool, execute_skim_qwen
from .focus import focus_tool, execute_focus
from .focus_qwen import focus_qwen_tool, execute_focus_qwen
from .localize_qwen import localize_qwen_tool, execute_localize_qwen
from .frame_verify import frame_verify_tool, execute_frame_verify
from .answer import answer_tool, execute_answer
from .registry import ToolRegistry


TOOLS = {
    "overview": overview_tool,
    "skim": skim_tool,
    "skim_qwen": skim_qwen_tool,
    "focus": focus_tool,
    "focus_qwen": focus_qwen_tool,
    "localize_qwen": localize_qwen_tool,
    "frame_verify": frame_verify_tool,
    "answer": answer_tool,
}


TOOL_FUNCTIONS = {
    "overview": execute_overview,
    "skim": execute_skim,
    "skim_qwen": execute_skim_qwen,
    "focus": execute_focus,
    "focus_qwen": execute_focus_qwen,
    "localize_qwen": execute_localize_qwen,
    "frame_verify": execute_frame_verify,
    "answer": execute_answer,
}


DEFAULT_TOOL_REGISTRY = ToolRegistry(tools=TOOLS, tool_functions=TOOL_FUNCTIONS)
