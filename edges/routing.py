"""
edges/routing.py

Routing functions for the LinkedIn agent graph.

HITL nodes (human_review_news, human_approval) route themselves via
Command(goto=...) - no functions needed for them.

Three functions remain:
  route_after_search     - error guard after search_news
  route_after_generation - error guard after generate_image
                           (image gen can fail if Colab is offline)
  route_after_publish    - error guard after publish_to_linkedin
"""

from __future__ import annotations

from langgraph.graph import END

from config.settings import WorkflowStatus
from core.state import BlueskyAgentState


def route_after_search(state: BlueskyAgentState) -> str:
    if state.get("workflow_status") == WorkflowStatus.ERROR:
        return "error_handler"
    return "human_review_news"


def route_after_generation(state: BlueskyAgentState) -> str:
    """Routes after generate_image. If image gen failed, go to error_handler."""
    if state.get("workflow_status") == WorkflowStatus.ERROR:
        return "error_handler"
    return "human_approval"


def route_after_publish(state: BlueskyAgentState) -> str:
    if state.get("workflow_status") in (WorkflowStatus.PUBLISH_FAILED, WorkflowStatus.ERROR):
        return "error_handler"
    return END