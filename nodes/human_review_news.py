"""
nodes/human_review_news.py

Node 2. HITL interrupt #1 -- news review.

How it works with the new interrupt/Command pattern:
  1. The node calls interrupt(), which immediately pauses the graph and
     surfaces the raw_news list to the caller. No external breakpoint config needed.
  2. The caller sees the articles, picks which ones to keep, then resumes
     by passing Command(resume={...}) back into graph.invoke/stream.
  3. The node re-executes from the top, interrupt() now returns the resume
     value instead of pausing, and we process the decision.

Resume payload the caller must send:
    Command(resume={
        "approved_indices": [0, 2, 4],   # list of ints  -> approved path
        "action": "approve"              # or "reject"   -> retry path
    })

The search_retry_count counter and max-retry cap are fully preserved.

Reads  : raw_news, search_retry_count
Writes : approved_news, news_review_status, search_retry_count, workflow_status
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.types import Command, interrupt

from config.settings import ReviewStatus, WorkflowConfig, WorkflowStatus, workflow_cfg
from core.state import TwitterAgentState

logger = logging.getLogger(__name__)


class HumanReviewNewsNode:
    """
    Pauses the graph so the human can review fetched news articles.

    Uses interrupt() to surface the articles to the caller and
    Command to route directly to the next node based on the decision,
    without needing any conditional edge after this node.
    """

    def __init__(self, wf_cfg: WorkflowConfig = workflow_cfg) -> None:
        self._wf_cfg = wf_cfg

    def __call__(
        self, state: TwitterAgentState
    ) -> Command[Literal["search_news", "generate_viral_hook", "generate_post_content", "error_handler"]]:

        raw_news    = state.get("raw_news") or []
        retry_count = state.get("search_retry_count", 0)

        # Pause here and send the article list to the caller.
        # interrupt() raises GraphInterrupt on the first pass.
        # On resume it returns whatever the caller put in Command(resume=...).
        decision = interrupt({
            "question": "Review the fetched news articles. Approve the ones you want to use.",
            "articles": [
                {
                    "index":  i,
                    "title":  item["title"],
                    "source": item["source"],
                    "url":    item["url"],
                }
                for i, item in enumerate(raw_news)
            ],
            "retry_count":  retry_count,
            "max_retries":  self._wf_cfg.max_search_retries,
            "instructions": (
                "Resume with Command(resume={'action': 'approve', 'approved_indices': [0, 2, 4]}) "
                "or Command(resume={'action': 'reject'}) to trigger a fresh search."
            ),
        })

        action           = decision.get("action", "approve")
        approved_indices = decision.get("approved_indices", [])

        logger.info(
            "HumanReviewNewsNode | action=%r | approved_indices=%s | retry=%d",
            action, approved_indices, retry_count,
        )

        # REJECT path -- bump counter, retry search or give up if cap reached
        if action == "reject" or not approved_indices:
            new_retry = retry_count + 1
            logger.info("HumanReviewNewsNode | rejected | retry_count now %d", new_retry)

            if new_retry >= self._wf_cfg.max_search_retries:
                logger.warning(
                    "HumanReviewNewsNode | max retries (%d) reached -> error_handler",
                    self._wf_cfg.max_search_retries,
                )
                return Command(
                    update={
                        "news_review_status": ReviewStatus.REJECTED,
                        "search_retry_count": new_retry,
                        "workflow_status":    WorkflowStatus.ERROR,
                        "error_message":      f"News rejected {new_retry} times -- max retries reached.",
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

        # APPROVE path -- build approved_news list from selected indices
        approved = [raw_news[i] for i in approved_indices if i < len(raw_news)]

        if not approved:
            # Indices were all out of range -- treat as a reject
            logger.warning("HumanReviewNewsNode | all approved_indices out of range -- treating as reject")
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

        # Fan-out to both parallel generation nodes using a list in goto
        return Command(
            update={
                "approved_news":      approved,
                "news_review_status": ReviewStatus.APPROVED,
                "workflow_status":    WorkflowStatus.NEWS_APPROVED,
            },
            goto=["generate_viral_hook", "generate_post_content"],
        )