"""
nodes/search_news.py

Node 1. Searches Tavily for recent news on the topic.

Reads  : topic, search_retry_count
Writes : raw_news, news_review_status, workflow_status, error_message
"""

from __future__ import annotations

import logging

from config.settings import ReviewStatus, WorkflowStatus
from core.state import BlueskyAgentState
from tools.search_tools import SearchTool

logger = logging.getLogger(__name__)


class SearchNewsNode:
    """
    Thin node - all search logic is in SearchTool.
    This class handles state in/out and error catching only.
    """

    def __init__(self, search_tool: SearchTool | None = None) -> None:
        self._tool = search_tool or SearchTool()

    def __call__(self, state: BlueskyAgentState) -> BlueskyAgentState:
        topic = state["topic"]
        logger.info("SearchNewsNode | topic=%r | retry=%d", topic, state.get("search_retry_count", 0))

        try:
            news_items = self._tool.search(topic)
            return {
                **state,
                "raw_news":           news_items,
                "news_review_status": ReviewStatus.PENDING,
                "workflow_status":    WorkflowStatus.NEWS_FETCHED,
                "error_message":      None,
            }
        except Exception as exc:
            logger.exception("SearchNewsNode | search failed")
            return {
                **state,
                "raw_news":        [],
                "error_message":   f"News search failed: {exc}",
                "workflow_status": WorkflowStatus.ERROR,
            }