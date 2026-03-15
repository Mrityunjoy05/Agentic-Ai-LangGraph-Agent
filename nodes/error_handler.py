"""
nodes/error_handler.py

Node 9. Catches workflow failures from any upstream node and logs
enough context to debug what went wrong.

We always end after this - no retries at this level. The MemorySaver
checkpoint captures the final state so you can inspect it after the run.

Reads  : error_message, workflow_status, topic, draft_version
Writes : workflow_status
"""

from __future__ import annotations

import logging

from config.settings import WorkflowStatus
from core.state import TwitterAgentState

logger = logging.getLogger(__name__)


class ErrorHandlerNode:
    """
    Logs the failure context and marks the workflow as handled so the
    graph ends cleanly rather than crashing mid-run.
    """

    def __call__(self, state: TwitterAgentState) -> TwitterAgentState:
        logger.error(
            "ErrorHandlerNode | WORKFLOW FAILED\n"
            "  topic          : %r\n"
            "  workflow_status: %r\n"
            "  error_message  : %r\n"
            "  draft_version  : %d",
            state.get("topic"),
            state.get("workflow_status"),
            state.get("error_message"),
            state.get("draft_version", 0),
        )

        return {
            **state,
            "workflow_status": WorkflowStatus.ERROR_HANDLED,
        }