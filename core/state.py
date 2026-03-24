"""
core/state.py

Shared state for the Bluesky Image Post Agent.

Every node reads from and returns a modified copy of this dict.
Nothing is mutated in place - LangGraph handles merging via MemorySaver.

Groups:
    INPUT      - set once by the caller, never changed by nodes
    SEARCH     - search_news and human_review_news
    GENERATION - hook node, content node, then image node
    APPROVAL   - human_approval (HITL #2)
    PUBLISH    - publish_to_bluesky
    SYSTEM     - error tracking and workflow routing
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class NewsItem(TypedDict):
    """One article returned by Tavily."""
    title: str
    url: str
    snippet: str       # trimmed to ~500 chars
    source: str        # domain only e.g. "techcrunch.com"
    published_at: str  # "recent" since Tavily doesn't return dates


class BlueskyPost(TypedDict):
    """The final assembled Bluesky post."""
    caption: str      # post text with hashtags
    char_count: int   # tracked so the human can see it at review time


class BlueskyAgentState(TypedDict):

    # INPUT
    topic: str
    target_audience: Optional[str]

    # SEARCH
    raw_news: Optional[List[NewsItem]]
    approved_news: Optional[List[NewsItem]]
    news_review_status: str      # pending -> approved | rejected
    search_retry_count: int

    # GENERATION
    hook: Optional[str]              # opening line, written by generate_hook
    post_caption: Optional[str]      # full post body, written by generate_content
    hashtags: Optional[List[str]]
    image_prompt: Optional[str]      # SDXL prompt, written by generate_content
    final_post: Optional[BlueskyPost]  # assembled caption, written by generate_content
    image_path: Optional[str]        # local file path, written by generate_image

    # APPROVAL
    approval_status: str    # pending -> approved | edited | rejected
    human_edits: Optional[str]

    # PUBLISH
    publish_receipt: Optional[dict]
    """
    {
      "post_uri": "at://did:plc:.../app.bsky.feed.post/...",
      "post_url": "https://bsky.app/profile/yourhandle/post/..."
    }
    """

    # SYSTEM
    error_message: Optional[str]
    workflow_status: str
    """
    started, news_fetched, news_approved, news_rejected_by_human,
    image_generated, approved_for_publish, human_edited,
    rejected_by_human, published_successfully, publish_failed,
    error, error_handled
    """