import hashlib
import re
from typing import Dict, Any, Optional


def _safe_function_id(function_id: Optional[str]) -> Optional[str]:
    if function_id is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(function_id)).strip("_")
    if len(cleaned) <= 64:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:53]}_{digest}"


class Action:
    """
    Represents an action with:
      - function_name (e.g. 'overview')
      - parameters    (a dictionary of parameter_name -> value)
    """

    def __init__(self, function_name: str, parameters: Dict[str, Any], function_id: Optional[str] = None):
        self.function_name = function_name
        self.parameters = parameters
        self.function_id = _safe_function_id(function_id)

    def __str__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, object]:
        return {"function": self.function_name, "parameters": self.parameters, "id": self.function_id}
