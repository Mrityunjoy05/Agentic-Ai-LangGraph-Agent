"""
nodes/human_approval.py

Node 5. HITL interrupt #2 -- final approval before publishing.

With content_optimization and post_validation removed, this node receives
final_draft directly from generate_post_content and shows it to the human.

The edit path now re-runs generate_post_content (via search_news -> human_review_news
would be too far back, so we go back to the parallel generation nodes instead).
Actually the cleanest approach: on edit, we write the instructions into state
and route back to generate_post_content so it regenerates with the instructions
added to the prompt context.

Resume options:
    Command(resume={"action": "approve"})
    Command(resume={"action": "edit", "instructions": "make tweet 3 shorter"})
    Command(resume={"action": "reject"})

Reads  : final_draft
Writes : approval_status, human_edits, workflow_status
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.types import Command, interrupt

from config.settings import ApprovalStatus, WorkflowStatus
from core.state import TwitterAgentState

logger = logging.getLogger(__name__)


class HumanApprovalNode:
    """
    Shows the assembled thread to the human and waits for their decision.
    No validation scores anymore since post_validation is removed -- just
    the raw draft with character counts so the human can judge for themselves.
    """

    def __call__(
        self, state: TwitterAgentState
    ) -> Command[Literal["publish_to_x", "generate_post_content", "__end__"]]:

        draft = state.get("final_draft") or []

        decision = interrupt({
            "question": "Review the final tweet draft. Approve, edit, or reject.",
            "draft": [
                {
                    "tweet_number":    t["tweet_number"],
                    "content":         t["content"],
                    "character_count": t["character_count"],
                }
                for t in draft
            ],
            "instructions": (
                "Resume with:\n"
                "  Command(resume={'action': 'approve'})\n"
                "  Command(resume={'action': 'edit', 'instructions': 'your edit notes here'})\n"
                "  Command(resume={'action': 'reject'})"
            ),
        })

        action       = decision.get("action", "approve")
        instructions = decision.get("instructions", "")

        logger.info(
            "HumanApprovalNode | action=%r | has_instructions=%s",
            action,
            bool(instructions),
        )

        if action == "approve":
            return Command(
                update={
                    "approval_status": ApprovalStatus.APPROVED,
                    "workflow_status": WorkflowStatus.APPROVED_FOR_PUBLISH,
                },
                goto="publish_to_x",
            )

        if action == "edit":
            if not instructions:
                # edit chosen but nothing written -- just approve
                logger.warning("HumanApprovalNode | edit chosen with no instructions -- auto-approving")
                return Command(
                    update={
                        "approval_status": ApprovalStatus.APPROVED,
                        "workflow_status": WorkflowStatus.APPROVED_FOR_PUBLISH,
                    },
                    goto="publish_to_x",
                )

            # Store the edit instructions and regenerate the content
            return Command(
                update={
                    "approval_status": ApprovalStatus.EDITED,
                    "human_edits":     instructions,
                    "workflow_status": WorkflowStatus.HUMAN_EDITED,
                },
                goto="generate_post_content",
            )

        # reject or anything unexpected
        logger.info("HumanApprovalNode | rejected -- ending workflow")
        return Command(
            update={
                "approval_status": ApprovalStatus.REJECTED,
                "workflow_status": WorkflowStatus.REJECTED_BY_HUMAN,
            },
            goto="__end__",
        )