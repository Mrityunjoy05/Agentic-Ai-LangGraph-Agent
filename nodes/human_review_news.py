"""
nodes/human_review_news.py

Node 2. HITL interrupt #1 - news review.

Graph pauses here. The human reviews raw_news and either approves
a subset or rejects all to trigger a fresh search.

Resume with:
    Command(resume={"action": "approve", "approved_indices": [0, 2]})
    Command(resume={"action": "reject"})

search_retry_count is preserved. After max_search_retries rejections
the graph routes to error_handler and ends.

Changed: on approve, routes to "generate_hook" only (sequential).
Previously routed to ["generate_hook", "generate_content"] in parallel.
Sequential is simpler and guarantees generate_content always has the
real hook available in state.

Reads  : raw_news, search_retry_count
Writes : approved_news, news_review_status, search_retry_count, workflow_status
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.types import Command, interrupt

from config.settings import ReviewStatus, WorkflowConfig, WorkflowStatus, workflow_cfg
from core.state import BlueskyAgentState

logger = logging.getLogger(__name__)


class HumanReviewNewsNode:

    def __init__(self, wf_cfg: WorkflowConfig = workflow_cfg) -> None:
        self._wf_cfg = wf_cfg

    def __call__(
        self, state: BlueskyAgentState
    ) -> Command[Literal["search_news", "generate_hook", "error_handler"]]:

        raw_news    = state.get("raw_news") or []
        retry_count = state.get("search_retry_count", 0)

        decision = interrupt({
            "question": "Review the fetched news articles. Pick which ones to use.",
            "articles": [
                {
                    "index":  i,
                    "title":  item["title"],
                    "source": item["source"],
                    "url":    item["url"],
                }
                for i, item in enumerate(raw_news)
            ],
            "retry_count": retry_count,
            "max_retries": self._wf_cfg.max_search_retries,
            "instructions": (
                "Resume with Command(resume={'action': 'approve', 'approved_indices': [0, 1, 2]}) "
                "or Command(resume={'action': 'reject'}) to trigger a fresh search."
            ),
        })

        action           = decision.get("action", "approve")
        approved_indices = decision.get("approved_indices", [])

        logger.info(
            "HumanReviewNewsNode | action=%r | indices=%s | retry=%d",
            action, approved_indices, retry_count,
        )

        # REJECT - bump counter, retry or give up
        if action == "reject" or not approved_indices:
            new_retry = retry_count + 1
            if new_retry >= self._wf_cfg.max_search_retries:
                return Command(
                    update={
                        "news_review_status": ReviewStatus.REJECTED,
                        "search_retry_count": new_retry,
                        "workflow_status":    WorkflowStatus.ERROR,
                        "error_message":      f"News rejected {new_retry} times - max retries reached.",
                    },
                    goto="error_handler",
                )
            return Command(
                update={
                    "news_review_status": ReviewStatus.REJECTED,
                    "search_retry_count": new_retry,
                    "workflow_status":    WorkflowStatus.NEWS_REJECTED,
                },
                goto="search_news",
            )

        # APPROVE - build approved list and route to generate_hook (sequential)
        approved = [raw_news[i] for i in approved_indices if i < len(raw_news)]
        if not approved:
            new_retry = retry_count + 1
            return Command(
                update={
                    "news_review_status": ReviewStatus.REJECTED,
                    "search_retry_count": new_retry,
                    "workflow_status":    WorkflowStatus.NEWS_REJECTED,
                },
                goto="search_news" if new_retry < self._wf_cfg.max_search_retries else "error_handler",
            )

        logger.info("HumanReviewNewsNode | approved %d/%d articles", len(approved), len(raw_news))
        return Command(
            update={
                "approved_news":      approved,
                "news_review_status": ReviewStatus.APPROVED,
                "workflow_status":    WorkflowStatus.NEWS_APPROVED,
            },
            goto="generate_hook",   # sequential - hook runs first, then content
        )