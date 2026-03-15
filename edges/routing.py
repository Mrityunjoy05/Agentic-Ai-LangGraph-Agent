"""
edges/routing.py

Conditional routing functions for the graph.

With content_optimization and post_validation removed, the graph is simpler.
Only two routing functions are needed now:

  route_after_search   -- error guard after search_news (Node 1)
  route_after_publish  -- error guard after publish_to_x (Node 6)

The HITL nodes (human_review_news, human_approval) still route themselves
via Command(goto=...) internally so no functions are needed for them here.

route_after_validation was removed along with post_validation.
"""

from __future__ import annotations

from langgraph.graph import END

from config.settings import WorkflowStatus
from core.state import TwitterAgentState


def route_after_search(state: TwitterAgentState) -> str:
    """Search errored -> error_handler. Succeeded -> human_review_news."""
    if state.get("workflow_status") == WorkflowStatus.ERROR:
        return "error_handler"
    return "human_review_news"


def route_after_publish(state: TwitterAgentState) -> str:
    """Publish failed -> error_handler for structured logging. Success -> END."""
    status = state.get("workflow_status")

    if status in (WorkflowStatus.PUBLISH_FAILED, WorkflowStatus.ERROR):
        return "error_handler"

    return END