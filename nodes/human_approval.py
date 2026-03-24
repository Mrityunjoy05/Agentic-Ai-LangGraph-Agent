"""
nodes/human_approval.py

Node 6. HITL interrupt #2 - final approval before posting to Bluesky.

The human sees:
  - The full post caption
  - The image path (so they can open it locally and check)
  - Character count

Resume with:
    Command(resume={"action": "approve"})
    Command(resume={"action": "edit", "instructions": "make the opening punchier"})
    Command(resume={"action": "reject"})

On edit: instructions go into state as human_edits, graph routes back
to generate_content which regenerates the post applying those instructions.
The image is also regenerated since the image_prompt may change.

Reads  : final_post, image_path
Writes : approval_status, human_edits, workflow_status
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.types import Command, interrupt

from config.settings import ApprovalStatus, WorkflowStatus
from core.state import BlueskyAgentState

logger = logging.getLogger(__name__)


class HumanApprovalNode:

    def __call__(
        self, state: BlueskyAgentState
    ) -> Command[Literal["publish_to_bluesky", "generate_content", "__end__"]]:

        final_post = state.get("final_post") or {}
        image_path = state.get("image_path") or "not generated yet"

        decision = interrupt({
            "question":    "Review the Bluesky post and image. Approve, edit, or reject.",
            "post": {
                "caption":    final_post.get("caption", ""),
                "char_count": final_post.get("char_count", 0),
            },
            "image_path":  image_path,
            "instructions": (
                "Resume with:\n"
                "  Command(resume={'action': 'approve'})\n"
                "  Command(resume={'action': 'edit', 'instructions': 'your notes here'})\n"
                "  Command(resume={'action': 'reject'})"
            ),
        })

        action       = decision.get("action", "approve")
        instructions = decision.get("instructions", "")

        logger.info("HumanApprovalNode | action=%r | has_instructions=%s", action, bool(instructions))

        if action == "approve":
            return Command(
                update={
                    "approval_status": ApprovalStatus.APPROVED,
                    "workflow_status": WorkflowStatus.APPROVED_FOR_PUBLISH,
                },
                goto="publish_to_bluesky",
            )

        if action == "edit":
            if not instructions:
                logger.warning("HumanApprovalNode | edit chosen with no instructions - auto-approving")
                return Command(
                    update={
                        "approval_status": ApprovalStatus.APPROVED,
                        "workflow_status": WorkflowStatus.APPROVED_FOR_PUBLISH,
                    },
                    goto="publish_to_bluesky",
                )
            return Command(
                update={
                    "approval_status": ApprovalStatus.EDITED,
                    "human_edits":     instructions,
                    "workflow_status": WorkflowStatus.HUMAN_EDITED,
                },
                goto="generate_content",
            )

        # reject
        logger.info("HumanApprovalNode | rejected - ending workflow")
        return Command(
            update={
                "approval_status": ApprovalStatus.REJECTED,
                "workflow_status": WorkflowStatus.REJECTED_BY_HUMAN,
            },
            goto="__end__",
        )