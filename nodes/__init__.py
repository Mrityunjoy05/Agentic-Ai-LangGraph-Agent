"""
nodes/__init__.py

Exposes all 8 node classes that make up the agent graph.

Graph order:
    1. SearchNewsNode          - fetch news via Tavily
    2. HumanReviewNewsNode     - HITL #1 - pick articles (interrupt/Command)
    3. GenerateHookNode        - Claude writes 3 hooks and scores them
    4. GenerateContentNode     - Claude writes post body + SDXL image prompt
    5. GenerateImageNode       - SDXL image generation via Colab Gradio
    6. HumanApprovalNode       - HITL #2 - review post + image (interrupt/Command)
    7. PublishToBlueskyNode    - post to Bluesky via atproto SDK
    8. ErrorHandlerNode        - graceful failure logging

Import examples:
    from nodes import SearchNewsNode, GenerateHookNode
    from nodes import HumanApprovalNode, PublishToBlueskyNode
"""

from nodes.error_handler import ErrorHandlerNode
from nodes.generate_content import GenerateContentNode
from nodes.generate_hook import GenerateHookNode
from nodes.generate_image import GenerateImageNode
from nodes.human_approval import HumanApprovalNode
from nodes.human_review_news import HumanReviewNewsNode
from nodes.publish_to_bluesky import PublishToBlueskyNode
from nodes.search_news import SearchNewsNode

__all__ = [
    "SearchNewsNode",
    "HumanReviewNewsNode",
    "GenerateHookNode",
    "GenerateContentNode",
    "GenerateImageNode",
    "HumanApprovalNode",
    "PublishToBlueskyNode",
    "ErrorHandlerNode",
]