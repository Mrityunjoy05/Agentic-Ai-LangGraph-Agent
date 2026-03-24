"""
core/__init__.py

Exposes the shared state TypedDicts that flow through every node in the graph.

Import examples:
    from core import BlueskyAgentState
    from core import NewsItem, BlueskyPost
"""

from core.state import BlueskyAgentState, BlueskyPost, NewsItem

__all__ = [
    "BlueskyAgentState",   # main state dict - every node reads and returns this
    "BlueskyPost",         # final assembled post (caption + char_count)
    "NewsItem",            # one article from Tavily search results
]