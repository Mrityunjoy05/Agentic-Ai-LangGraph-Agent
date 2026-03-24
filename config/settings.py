"""
config/settings.py

All config and constants for the LinkedIn Image Post Agent.
Change things here and every file picks it up automatically.

Keys are read from .env at import time. Missing key = KeyError immediately,
which is much better than a silent failure three nodes later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class GroqConfig:
    api_key: str = os.getenv('GROQ_API_KEY')
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 4096
    temperature: float = 0


@dataclass(frozen=True)
class TavilyConfig:
    # TavilySearch reads TAVILY_API_KEY from env automatically
    max_results: int = 5
    search_depth: str = "advanced"
    include_answer: bool = True
    include_raw_content: bool = False


@dataclass(frozen=True)
class ImageGenConfig:
    # The Gradio public URL from your running Google Colab cell
    # e.g. "https://1234abcd.gradio.live"
    gradio_url: str = os.getenv("GRADIO_URL")
    negative_prompt: str = (
        "cartoon, painting, illustration, blurry, deformed, ugly, "
        "low quality, CGI, toy, fake, animated, artificial background"
    )
    steps: int = 50
    guidance: float = 7.5
    save_dir: str = "data"          # relative to project root, created if missing


@dataclass(frozen=True)
class BlueskyConfig:
    # Your Bluesky handle e.g. "yourname.bsky.social"
    handle: str = os.getenv('BLUESKY_HANDLE')
    # App password generated from Settings -> Privacy and Security -> App Passwords
    app_password: str = os.getenv("BLUESKY_APP_PASSWORD")
    # Bluesky API base URL - this never changes
    api_url: str = "https://bsky.social"


@dataclass(frozen=True)
class WorkflowConfig:
    max_search_retries: int = 3
    hook_variants_count: int = 3
    # Bluesky post character limit is 300 per post
    max_post_chars: int = 290
    graph_thread_id_prefix: str = "bluesky_agent_run"


class WorkflowStatus:
    STARTED                = "started"
    NEWS_FETCHED           = "news_fetched"
    NEWS_APPROVED          = "news_approved"
    NEWS_REJECTED          = "news_rejected_by_human"
    IMAGE_GENERATED        = "image_generated"
    APPROVED_FOR_PUBLISH   = "approved_for_publish"
    HUMAN_EDITED           = "human_edited"
    REJECTED_BY_HUMAN      = "rejected_by_human"
    PUBLISHED_SUCCESSFULLY = "published_successfully"
    PUBLISH_FAILED         = "publish_failed"
    ERROR                  = "error"
    ERROR_HANDLED          = "error_handled"


class ReviewStatus:
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalStatus:
    PENDING  = "pending"
    APPROVED = "approved"
    EDITED   = "edited"
    REJECTED = "rejected"


groq_cfg  = GroqConfig()
tavily_cfg     = TavilyConfig()
image_gen_cfg  = ImageGenConfig()
bluesky_cfg    = BlueskyConfig()
workflow_cfg   = WorkflowConfig()