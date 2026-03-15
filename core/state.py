"""
core/state.py

This is the heart of the whole project - the shared state that every node
reads from and writes to. Keep it clean.

A few things I learned the hard way when setting this up:
- Keep sub-structures as plain TypedDicts (not dataclasses). MemorySaver
  handles serialisation without any extra work that way.
- Mark fields Optional when the node that owns them hasn't run yet.
  Otherwise LangGraph complains at graph.compile() time.
- Don't let nodes touch fields they don't own. It makes debugging a nightmare
  when something goes wrong mid-run.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


# These two are the building blocks. Keeping them typed saves a lot of
# dict.get() guessing in the nodes.

class NewsItem(TypedDict):
    """One article from the Tavily search results."""
    title: str
    url: str
    snippet: str          # trimmed to ~500 chars - just enough context
    source: str           # domain only, e.g. "techcrunch.com"
    published_at: str     # whatever Tavily returns; sometimes just "recent"


class TweetThread(TypedDict):
    """Represents a single tweet, whether part of a thread or standalone."""
    tweet_number: int
    content: str
    character_count: int  # tracked separately so we don't recount later


# The main state object. Fields are grouped by which part of the workflow
# owns them - makes it much easier to trace where a bug came from.

class TwitterAgentState(TypedDict):
    """
    Shared state across the whole graph. Every node gets this dict and
    returns a modified copy - nothing mutated in place.

    Groups:
        INPUT      - set by the caller at start(), never changed after that
        SEARCH     - search_news and human_review_news
        GENERATION - the two parallel gen nodes (also owns final assembly)
        APPROVAL   - human_approval (HITL #2)
        PUBLISH    - publish_to_x
        SYSTEM     - any node can write here for status/error tracking
    """

    # INPUT - caller sets these once, nodes should treat them as read-only
    topic: str
    """The subject to search news for, e.g. 'AI Agents 2026'."""

    content_type: str
    """'single_tweet' or 'thread'."""

    target_audience: Optional[str]
    """Who we're writing for. Passed straight into the LLM prompts."""

    schedule_time: Optional[str]
    """ISO-8601 string if the user wants to schedule, None means post now."""

    media_urls: Optional[List[str]]
    """Up to 4 image/video URLs to attach. X's limit, not ours."""

    # SEARCH - search_news writes raw_news, human_review_news writes approved_news
    raw_news: Optional[List[NewsItem]]
    """Everything Tavily found. Human picks from these."""

    approved_news: Optional[List[NewsItem]]
    """The subset the human actually wants to use."""

    news_review_status: str
    """'pending' -> 'approved' or 'rejected'. Drives the HITL #1 edge."""

    search_retry_count: int
    """Tracks how many times the human has rejected news. We stop at 3."""

    # GENERATION - generate_viral_hook and generate_post_content run in parallel.
    # generate_post_content also owns final_draft since optimization is gone.
    viral_hook: Optional[str]
    """The opening tweet. generate_viral_hook picks the best of 3 options."""

    post_content: Optional[List[TweetThread]]
    """Raw body tweets (2 onwards) before assembly. Kept for reference."""

    hashtags: Optional[List[str]]
    """Five hashtags appended to the last tweet."""

    cta: Optional[str]
    """Call-to-action line at the end of the thread."""

    final_draft: Optional[List[TweetThread]]
    """Fully assembled, numbered thread. Written by generate_post_content,
    shown to the human at HITL #2."""

    # APPROVAL - human fills these in during HITL #2
    approval_status: str
    """'pending' -> 'approved', 'edited', or 'rejected'."""

    human_edits: Optional[str]
    """Edit instructions from the human during HITL #2. generate_post_content
    reads this on re-run and clears it after applying."""

    # PUBLISH
    publish_receipt: Optional[dict]
    """
    What comes back from the X API after posting:
    {
      "thread_url": "https://twitter.com/user/status/123",
      "tweets": [{"tweet_number": 1, "tweet_id": "123", "url": "..."}]
    }
    """

    # SYSTEM - used for routing and observability throughout
    error_message: Optional[str]
    """Set by any node that catches an exception. Logged by error_handler."""

    workflow_status: str
    """
    Tracks where we are. Routing functions in edges/routing.py read this.
    Values: started, news_fetched, news_approved, news_rejected_by_human,
    approved_for_publish, human_edited, rejected_by_human,
    published_successfully, publish_failed, error, error_handled
    """