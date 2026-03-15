"""
nodes/publish_to_x.py

Node 8. Posts the approved content to X.

Delegates everything to XPublisher so this node stays thin. The only
logic here is catching Tweepy exceptions and translating them into
state fields that error_handler can log properly.

Reads  : final_draft, media_urls, schedule_time
Writes : publish_receipt, workflow_status, error_message
"""

from __future__ import annotations

import logging

import tweepy

from config.settings import WorkflowStatus
from core.state import TwitterAgentState
from tools.x_api_tools import XPublisher

logger = logging.getLogger(__name__)


class PublishToXNode:
    """
    Calls XPublisher and stores the receipt in state. Pass a mock publisher
    in tests so you don't accidentally post to X during development.
    """

    def __init__(self, publisher: XPublisher | None = None) -> None:
        self._publisher = publisher or XPublisher()

    #
    def __call__(self, state: TwitterAgentState) -> TwitterAgentState:
        tweets     = state.get("final_draft") or []
        media_urls = state.get("media_urls")

        logger.info("PublishToXNode | tweets=%d | media=%s", len(tweets), bool(media_urls))

        try:
            receipt = self._publisher.publish(tweets=tweets, media_urls=media_urls)

            return {
                **state,
                "publish_receipt": receipt,
                "workflow_status": WorkflowStatus.PUBLISHED_SUCCESSFULLY,
                "error_message": None,
            }

        except tweepy.TweepyException as exc:
            logger.exception("PublishToXNode | X API error")
            return {
                **state,
                "publish_receipt": None,
                "error_message": f"X API error: {exc}",
                "workflow_status": WorkflowStatus.PUBLISH_FAILED,
            }

        except Exception as exc:
            logger.exception("PublishToXNode | unexpected error")
            return {
                **state,
                "publish_receipt": None,
                "error_message": f"Unexpected publish error: {exc}",
                "workflow_status": WorkflowStatus.ERROR,
            }