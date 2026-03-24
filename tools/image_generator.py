"""
tools/image_generator.py

Sends an image generation request to the SDXL model running on
Google Colab via the Gradio public URL, saves the result locally,
and returns the file path.

How the setup works:
  1. You run the Colab notebook - it launches a Gradio server and
     prints a public URL like https://1234abcd.gradio.live
  2. You put that URL in your .env as GRADIO_URL
  3. This class calls that Gradio endpoint using gradio_client,
     which sends the prompt over HTTP and gets back the image file
  4. The image is saved to data/ and the path is returned

The generate() method is intentionally simple - just a prompt in,
file path out. All the SDXL parameters (steps, guidance, negative
prompt) come from ImageGenConfig so you only tune them in one place.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from gradio_client import Client

from config.settings import ImageGenConfig, image_gen_cfg

logger = logging.getLogger(__name__)


class ImageGeneratorTool:
    """
    Wraps the Gradio client that talks to SDXL running on Colab GPU.

    Pass a different cfg if you want different generation settings.
    The Gradio URL is read from cfg.gradio_url which comes from
    the GRADIO_URL environment variable.
    """

    def __init__(self, cfg: ImageGenConfig = image_gen_cfg) -> None:
        self._cfg = cfg
        self._client = Client(cfg.gradio_url)

        # Make sure the save directory exists
        self.image_folder_path = Path(__file__).parent.parent / cfg.save_dir
        Path(self.image_folder_path).mkdir(parents=True, exist_ok=True)

        logger.info("ImageGeneratorTool | connected to Gradio at %s", cfg.gradio_url)

    def generate(self, prompt: str) -> str:
        """
        Generate an image from the given SDXL prompt.

        Returns the local file path where the image was saved.
        Raises RuntimeError if generation or saving fails.

        The file is saved as data/<uuid>.png so each run gets
        a unique filename and old images don't get overwritten.
        """
        logger.info("ImageGeneratorTool.generate | sending prompt (%d chars)", len(prompt))
        logger.info("ImageGeneratorTool.generate | prompt: %r", prompt[:100])

        try:
            result = self._client.predict(
                prompt,
                self._cfg.negative_prompt,
                self._cfg.steps,
                self._cfg.guidance,
                api_name="/predict",
            )

            # result is the file path Gradio returns (temp file on Gradio server)
            # We save our own copy with a stable name
            save_path = os.path.join(self.image_folder_path, f"{uuid.uuid4().hex}.png")

            from PIL import Image
            image = Image.open(result)
            image = image.resize((900, 900), Image.Resampling.LANCZOS)
            image.save(save_path)

            size_kb = os.path.getsize(save_path) / 1024
            logger.info(
                "ImageGeneratorTool.generate | saved to %s (%.1f KB)",
                save_path, size_kb,
            )
            return save_path

        except Exception as exc:
            logger.exception("ImageGeneratorTool.generate | failed")
            raise RuntimeError(f"Image generation failed: {exc}") from exc