"""
tools/x_api_tools.py

Handles everything X/Twitter API related. There's one annoying wrinkle:
media uploads still require the v1.1 API because v2 doesn't support them
yet. So we keep both clients around - v2 for posting, v1.1 just for media.

The 0.5s sleep between tweets in a thread isn't strictly required but in
practice hitting the API in rapid succession for long threads occasionally
triggers rate limit errors, so I kept it.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import tweepy

from config.settings import WorkflowConfig, XAPIConfig, workflow_cfg, x_api_cfg
from core.state import TweetThread

logger = logging.getLogger(__name__)


class XPublisher:
    """
    Posts tweets and threads to X via Tweepy. Handles media uploads too.

    Inject a different cfg or wf_cfg if you need to override settings
    per-run. In tests, just pass a mock instance to PublishToXNode directly.
    """

    def __init__(
        self,
        cfg: XAPIConfig = x_api_cfg,
        wf_cfg: WorkflowConfig = workflow_cfg,
    ) -> None:
        self._cfg = cfg
        self._wf_cfg = wf_cfg

        # v2 client for creating tweets and reply chains
        self._client_v2 = tweepy.Client(
            bearer_token=cfg.bearer_token,
            consumer_key=cfg.api_key,
            consumer_secret=cfg.api_secret,
            access_token=cfg.access_token,
            access_token_secret=cfg.access_token_secret,
            wait_on_rate_limit=True,
        )

        # v1.1 only used for media uploads - v2 doesn't support this yet
        _auth = tweepy.OAuthHandler(cfg.api_key, cfg.api_secret)
        _auth.set_access_token(cfg.access_token, cfg.access_token_secret)
        self._api_v1 = tweepy.API(_auth, wait_on_rate_limit=True)

    def publish(
        self,
        tweets: List[TweetThread],
        media_urls: Optional[List[str]] = None,
    ) -> Dict:
        """
        Post a single tweet or a full thread. For threads, each tweet is
        posted as a reply to the previous one to create the chain.

        Returns a receipt dict with all the tweet IDs and URLs.
        Raises tweepy.TweepyException on API errors.
        """
        media_ids = self._upload_media(media_urls) if media_urls else []

        receipt: Dict = {"tweets": [], "thread_url": ""}
        prev_tweet_id: Optional[str] = None

        for idx, tweet in enumerate(tweets):
            kwargs: Dict = {"text": tweet["content"]}

            if prev_tweet_id:
                kwargs["in_reply_to_tweet_id"] = prev_tweet_id

            # Only attach media to the first tweet
            if media_ids and idx == 0:
                kwargs["media_ids"] = media_ids[: self._wf_cfg.max_media_attachments]

            logger.info(
                "XPublisher.publish | tweet %d/%d | chars=%d",
                idx + 1,
                len(tweets),
                tweet["character_count"],
            )

            response = self._client_v2.create_tweet(**kwargs)
            tweet_id = str(response.data["id"])
            prev_tweet_id = tweet_id

            receipt["tweets"].append(
                {
                    "tweet_number": tweet["tweet_number"],
                    "tweet_id": tweet_id,
                    "url": f"https://twitter.com/i/web/status/{tweet_id}",
                }
            )

            # Brief pause to avoid hitting burst limits on longer threads
            if idx < len(tweets) - 1:
                time.sleep(0.5)

        if receipt["tweets"]:
            receipt["thread_url"] = receipt["tweets"][0]["url"]

        logger.info("XPublisher.publish | done | thread_url=%s", receipt["thread_url"])
        return receipt

    def _upload_media(self, media_urls: List[str]) -> List[int]:
        """
        Upload each URL via v1.1 and collect the media_ids.
        Skips any URL that fails rather than aborting the whole publish.
        """
        media_ids: List[int] = []
        cap = self._wf_cfg.max_media_attachments

        for url in media_urls[:cap]:
            try:
                media = self._api_v1.media_upload(url)
                media_ids.append(media.media_id)
                logger.info("XPublisher._upload_media | uploaded %s -> %s", url, media.media_id)
            except tweepy.TweepyException as exc:
                logger.warning("XPublisher._upload_media | skipped %s - %s", url, exc)

        return media_ids