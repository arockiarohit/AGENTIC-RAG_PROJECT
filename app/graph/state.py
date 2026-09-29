from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict):
    question: str
    session_id: str
    route: str
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    image_path: str | None