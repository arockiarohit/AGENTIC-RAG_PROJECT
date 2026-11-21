from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict):
    original_query: str
    normalized_query: str
    language: str
    question: str
    session_id: str
    route: str
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    image_path: str | None
    location: str | None
    time_range: str | None
    audit_logs: List[Dict[str, Any]] | None