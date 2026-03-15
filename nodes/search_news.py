"""
nodes/search_news.py

Node 1. Searches Tavily for recent news on the topic and stores
the results in state["raw_news"] for the human to review.

Reads  : topic, search_retry_count
Writes : raw_news, news_review_status, workflow_status, error_message
"""

from __future__ import annotations

import logging

from config.settings import ReviewStatus, WorkflowStatus
from core.state import TwitterAgentState
from tools.search_tools import SearchTool

logger = logging.getLogger(__name__)


class SearchNewsNode:
    """
    Keeps the node thin - all the actual search logic is in SearchTool.
    This class just handles state in/out and error catching.

    Pass in a search_tool for testing so you're not hitting Tavily in unit tests.
    """

    def __init__(self, search_tool: SearchTool | None = None) -> None:
        self._tool = search_tool or SearchTool()

    def __call__(self, state: TwitterAgentState) -> TwitterAgentState:
        """
        Runs every time LangGraph visits this node, including on retries
        after the human rejects the news results.
        """
        topic = state["topic"]
        logger.info("SearchNewsNode | topic=%r | retry=%d", topic, state.get("search_retry_count", 0))

        try:
            news_items = self._tool.search(topic)

            return {
                **state,
                "raw_news": news_items,
                "news_review_status": ReviewStatus.PENDING,
                "workflow_status": WorkflowStatus.NEWS_FETCHED,
                "error_message": None,
            }

        except Exception as exc:
            logger.exception("SearchNewsNode | search failed")
            return {
                **state,
                "raw_news": [],
                "error_message": f"News search failed: {exc}",
                "workflow_status": WorkflowStatus.ERROR,
            }