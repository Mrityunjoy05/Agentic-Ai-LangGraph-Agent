"""
nodes/error_handler.py

Safety net. Any node that sets workflow_status = 'error' routes here.
Logs the failure context and ends the graph cleanly so the MemorySaver
checkpoint captures the final state for inspection.

Reads  : error_message, workflow_status, topic
Writes : workflow_status
"""

from __future__ import annotations

import logging

from config.settings import WorkflowStatus
from core.state import BlueskyAgentState

logger = logging.getLogger(__name__)


class ErrorHandlerNode:

    def __call__(self, state: BlueskyAgentState) -> BlueskyAgentState:
        logger.error(
            "ErrorHandlerNode | WORKFLOW FAILED\n"
            "  topic          : %r\n"
            "  workflow_status: %r\n"
            "  error_message  : %r",
            state.get("topic"),
            state.get("workflow_status"),
            state.get("error_message"),
        )
        return {
            **state,
            "workflow_status": WorkflowStatus.ERROR_HANDLED,
        }