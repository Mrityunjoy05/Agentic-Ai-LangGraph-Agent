"""
tools/bluesky_tools.py

Posts to Bluesky using the official atproto Python SDK.

Using atproto instead of raw requests means:
  - No manual JWT handling
  - No manually building headers
  - No manually formatting timestamps
  - The Client object handles session, token refresh, and all API details

Install:
    pip install atproto
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from atproto import Client

from config.settings import BlueskyConfig, bluesky_cfg

logger = logging.getLogger(__name__)


class BlueskyPublisher:
    """
    Posts an image + caption to Bluesky using the atproto Client.

    A fresh login is done on every publish() call so token expiry
    is never an issue. Pass a mock instance in tests.
    """

    def __init__(self, cfg: BlueskyConfig = bluesky_cfg) -> None:
        self._cfg = cfg

    def publish(self, caption: str, image_path: str) -> Dict:
        """
        Login, upload the image, and create the post.

        Returns:
        {
          "post_uri": "at://did:plc:.../app.bsky.feed.post/...",
          "post_url": "https://bsky.app/profile/yourhandle/post/..."
        }

        Raises Exception on any API failure.
        """
        # Fresh login every time - no stored tokens to expire
        client = Client()
        client.login(self._cfg.handle, self._cfg.app_password)
        logger.info("BlueskyPublisher | logged in as %s", self._cfg.handle)

        # Upload the image and get back a blob reference
        image_bytes = Path(image_path).read_bytes()
        blob_response = client.upload_blob(image_bytes)
        logger.info("BlueskyPublisher | image uploaded (%d bytes)", len(image_bytes))

        # Create the post with the image attached
        from atproto import models

        post_response = client.send_post(
            text=caption,
            embed=models.AppBskyEmbedImages.Main(
                images=[
                    models.AppBskyEmbedImages.Image(
                        image=blob_response.blob,
                        alt=caption[:100],
                    )
                ]
            ),
        )

        rkey     = post_response.uri.split("/")[-1]
        post_url = f"https://bsky.app/profile/{self._cfg.handle}/post/{rkey}"

        receipt = {
            "post_uri": post_response.uri,
            "post_url": post_url,
        }
        logger.info("BlueskyPublisher | posted | url=%s", post_url)
        return receipt