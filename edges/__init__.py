"""
edges/__init__.py

Exposes the conditional routing functions used in main.py to wire
the graph edges.

The two HITL nodes (human_review_news, human_approval) route themselves
via Command(goto=...) internally so they do not need routing functions here.

Import examples:
    from edges import route_after_search
    from edges import route_after_generation, route_after_publish
"""

from edges.routing import (
    route_after_generation,
    route_after_publish,
    route_after_search,
)

__all__ = [
    "route_after_search",      # error guard after search_news (Node 1)
    "route_after_generation",  # error guard after generate_image (Node 5)
    "route_after_publish",     # error guard after publish_to_bluesky (Node 7)
]