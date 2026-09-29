from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState

from app.graph.nodes import (
    theni_retriever_node,
    web_search_node,
    calculator_node,
    generate_answer_node,
    image_generation_node
)

from app.graph.router import classify_question, route_question

import logging

logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:

    workflow = StateGraph(GraphState)

    # =========================
    # Add Nodes
    # =========================

    workflow.add_node("classify", classify_question)

    workflow.add_node(
        "theni_retriever",
        theni_retriever_node
    )

    workflow.add_node(
        "web_search",
        web_search_node
    )

    workflow.add_node(
        "calculator",
        calculator_node
    )

    workflow.add_node(
        "image_generation",
        image_generation_node
    )

    workflow.add_node(
        "generate_answer",
        generate_answer_node
    )

    # =========================
    # START
    # =========================

    workflow.add_edge(
        START,
        "classify"
    )

    # =========================
    # ROUTING
    # =========================

    workflow.add_conditional_edges(
        "classify",
        route_question,
        {
            "theni": "theni_retriever",
            "general": "web_search",
            "current": "web_search",
            "arithmetic": "calculator",
            "image": "image_generation"
        }
    )

    # =========================
    # Normal RAG / Web Flow
    # =========================

    workflow.add_edge(
        "theni_retriever",
        "generate_answer"
    )

    workflow.add_edge(
        "web_search",
        "generate_answer"
    )

    workflow.add_edge(
        "calculator",
        "generate_answer"
    )

    # =========================
    # Image Flow
    # =========================

    workflow.add_edge(
        "image_generation",
        END
    )

    # =========================
    # Final Answer
    # =========================

    workflow.add_edge(
        "generate_answer",
        END
    )

    logger.info(
        "LangGraph workflow compiled successfully."
    )

    return workflow.compile()


# Global compiled graph
graph = create_workflow()