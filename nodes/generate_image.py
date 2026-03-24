"""
nodes/generate_image.py

Node 5. Runs after both parallel generation nodes finish.

Takes the image_prompt from generate_content and sends it to the
SDXL model running on Google Colab via the Gradio client.
Saves the image locally and writes the file path to state.

If image generation fails (Colab disconnected, bad URL, etc.) the node
catches the error, sets workflow_status = error, and lets error_handler
deal with it cleanly rather than crashing the whole graph.

Reads  : image_prompt
Writes : image_path, workflow_status, error_message
"""

from __future__ import annotations

import logging

from config.settings import WorkflowStatus
from core.state import BlueskyAgentState
from tools.image_generator import ImageGeneratorTool

logger = logging.getLogger(__name__)


class GenerateImageNode:
    """
    Calls ImageGeneratorTool with the SDXL prompt from generate_content
    and writes the saved image path back to state.

    Pass a mock tool in tests so you're not hitting the Colab GPU.
    """

    def __init__(self, image_tool: ImageGeneratorTool | None = None) -> None:
        self._tool = image_tool or ImageGeneratorTool()

    def __call__(self, state: BlueskyAgentState) -> BlueskyAgentState:
        image_prompt = state.get("image_prompt")

        if not image_prompt:
            logger.error("GenerateImageNode | no image_prompt in state")
            return {
                **state,
                "error_message":   "generate_content did not produce an image_prompt",
                "workflow_status": WorkflowStatus.ERROR,
            }

        logger.info("GenerateImageNode | generating image | prompt=%r", image_prompt[:80])

        try:
            image_path = self._tool.generate(image_prompt)
            logger.info("GenerateImageNode | image saved at %s", image_path)

            return {
                **state,
                "image_path":      image_path,
                "workflow_status": WorkflowStatus.IMAGE_GENERATED,
                "error_message":   None,
            }

        except Exception as exc:
            logger.exception("GenerateImageNode | image generation failed")
            return {
                **state,
                "image_path":      None,
                "error_message":   f"Image generation failed: {exc}",
                "workflow_status": WorkflowStatus.ERROR,
            }