"""
tools/__init__.py

Exposes the three tool classes used by the nodes.

Each tool wraps an external service and keeps all service-specific
logic out of the nodes. Nodes only handle state in/out.

Tools:
    SearchTool          - wraps Tavily search, returns List[NewsItem]
    ImageGeneratorTool  - wraps Gradio/SDXL on Colab, returns image file path
    BlueskyPublisher    - wraps atproto SDK, posts image + caption to Bluesky

Import examples:
    from tools import SearchTool
    from tools import ImageGeneratorTool, BlueskyPublisher
"""

from tools.bluesky_tools import BlueskyPublisher
from tools.image_generator import ImageGeneratorTool
from tools.search_tools import SearchTool

__all__ = [
    "BlueskyPublisher",    # used by PublishToBlueskyNode
    "ImageGeneratorTool",  # used by GenerateImageNode
    "SearchTool",          # used by SearchNewsNode
]