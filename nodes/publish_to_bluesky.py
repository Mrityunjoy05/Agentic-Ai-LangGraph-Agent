"""
nodes/publish_to_bluesky.py

Node 7. Posts the approved image + caption to Bluesky.

Delegates to BlueskyPublisher so this node stays thin.
Catches requests.HTTPError from the Bluesky API and stores
it cleanly in state rather than crashing the graph.

Reads  : final_post, image_path, hashtags
Writes : publish_receipt, workflow_status, error_message
"""

from __future__ import annotations

import logging

import requests

from config.settings import WorkflowStatus
from core.state import BlueskyAgentState
from tools.bluesky_tools import BlueskyPublisher

logger = logging.getLogger(__name__)


class PublishToBlueskyNode:
    """
    Calls BlueskyPublisher and stores the post receipt in state.
    Pass a mock publisher in tests so you don't accidentally post during development.
    """

    def __init__(self, publisher: BlueskyPublisher | None = None) -> None:
        self._publisher = publisher or BlueskyPublisher()

    def __call__(self, state: BlueskyAgentState) -> BlueskyAgentState:
        final_post = state.get("final_post") or {}
        image_path = state.get("image_path")

        # Reassemble caption with hashtags appended
        caption  = final_post.get("caption", "")
        hashtags = state.get("hashtags") or []
        if hashtags:
            caption = f"{caption}\n\n{' '.join(hashtags)}"

        # Bluesky has a 300 char limit - trim if needed (shouldn't happen if prompts are tuned)
        if len(caption) > 300:
            caption = caption[:297] + "..."

        logger.info(
            "PublishToBlueskyNode | caption=%d chars | image=%s",
            len(caption), image_path,
        )

        if not image_path:
            return {
                **state,
                "error_message":   "No image_path in state - image generation may have failed.",
                "workflow_status": WorkflowStatus.ERROR,
            }

        try:
            receipt = self._publisher.publish(caption=caption, image_path=image_path)
            return {
                **state,
                "publish_receipt": receipt,
                "workflow_status": WorkflowStatus.PUBLISHED_SUCCESSFULLY,
                "error_message":   None,
            }

        except requests.HTTPError as exc:
            logger.exception("PublishToBlueskyNode | Bluesky API error")
            return {
                **state,
                "publish_receipt": None,
                "error_message":   f"Bluesky API error: {exc}",
                "workflow_status": WorkflowStatus.PUBLISH_FAILED,
            }

        except Exception as exc:
            logger.exception("PublishToBlueskyNode | unexpected error")
            return {
                **state,
                "publish_receipt": None,
                "error_message":   f"Unexpected publish error: {exc}",
                "workflow_status": WorkflowStatus.ERROR,
            }