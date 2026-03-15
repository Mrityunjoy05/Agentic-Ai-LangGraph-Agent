"""
config/settings.py

All config and constants in one place. The main reason for this file is
that I got tired of hunting down hardcoded strings and numbers scattered
across nodes - if something needs tuning (retry cap, model name, thread
length), you change it here and nowhere else.

All API keys are read from environment variables at import time. If a
required key is missing you'll get a KeyError immediately on startup,
which is much better than a cryptic failure mid-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv()


# API credentials - frozen so nothing accidentally mutates them at runtime

@dataclass(frozen=True)
class GroqConfig:
    api_key: str = os.getenv('GROQ_API_KEY')
    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 4096
    temperature: float = 0.7

@dataclass(frozen=True)
class TavilyConfig:
    # TavilySearch reads TAVILY_API_KEY from the environment automatically,
    # so no need to pass the key here unlike the old TavilySearchResults.
    max_results: int = 8
    search_depth: str = "advanced"     # "basic" is faster but misses a lot
    include_answer: bool = True
    include_raw_content: bool = False


# @dataclass(frozen=True)
# class XAPIConfig:
#     bearer_token: str = field(default_factory=lambda: os.environ["X_BEARER_TOKEN"])
#     api_key: str = field(default_factory=lambda: os.environ["X_API_KEY"])
#     api_secret: str = field(default_factory=lambda: os.environ["X_API_SECRET"])
#     access_token: str = field(default_factory=lambda: os.environ["X_ACCESS_TOKEN"])
#     access_token_secret: str = field(default_factory=lambda: os.environ["X_ACCESS_TOKEN_SECRET"])


# Workflow behaviour - tweak these to adjust how the agent behaves

@dataclass(frozen=True)
class WorkflowConfig:
    # How many times the human can reject news before we give up
    max_search_retries: int = 3

    # LLM generates this many hook options before picking the best
    hook_variants_count: int = 3

    min_thread_tweets: int = 5
    max_thread_tweets: int = 8
    max_media_attachments: int = 4     # X only allows 4 per tweet

    graph_thread_id_prefix: str = "twitter_agent_run"


# Status strings - used by nodes and routing functions.
# Having them here means you rename in one place and every file gets it.

class WorkflowStatus:
    STARTED                = "started"
    NEWS_FETCHED           = "news_fetched"
    NEWS_APPROVED          = "news_approved"
    NEWS_REJECTED          = "news_rejected_by_human"
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


# Module-level singletons - import these directly instead of instantiating
# new config objects in each file.

groq_cfg  = GroqConfig()
tavily_cfg     = TavilyConfig()
# x_api_cfg      = XAPIConfig()
workflow_cfg   = WorkflowConfig()