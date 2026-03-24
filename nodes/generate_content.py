"""
nodes/generate_content.py

Node 4. Runs after generate_hook (sequential, not parallel anymore).

Receives the hook directly from state - no placeholder needed since
generate_hook always runs first now.

Generates:
  1. The Bluesky post body
  2. An SDXL image prompt - detailed, cinematic style with technical
     quality tags at the end (photorealistic, cinematic, 8k)

Changes from the previous version:
  - Hook is always available in state (no parallel race condition)
  - IMAGE_PROMPT format updated: detailed scene description ending with
    photorealistic, cinematic, 8k style tags
  - _parse_response fixed: hashtags are now extracted more robustly.
    The old approach only caught hashtags that were on their own clean line.
    The new approach scans all lines for hashtag tokens (#word) regardless
    of whether they're mixed with other text, which handles Claude's
    tendency to put them inline or on a partially-formatted line.

Reads  : approved_news, topic, target_audience, hook, human_edits
Writes : post_caption, hashtags, image_prompt, final_post, human_edits (cleared)
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from langchain_groq import ChatGroq

from config.settings import GroqConfig, WorkflowConfig, groq_cfg, workflow_cfg
from core.state import BlueskyAgentState, BlueskyPost

logger = logging.getLogger(__name__)


_CONTENT_PROMPT = """\
You are a professional Bluesky content strategist.

Topic: {topic}
Target Audience: {audience}
Opening Hook (already written - copy this exactly as your first line): {hook}

News Articles (base ALL content on these - do NOT invent facts):
{news_context}

Write a Bluesky post AND a detailed SDXL image prompt.

--- POST REQUIREMENTS ---
- First line must be the hook above, copied exactly
- 100-250 words total
- 2-3 short paragraphs separated by blank lines
- Each paragraph adds one insight or piece of evidence from the news articles
- End with a thought-provoking question or a clear call-to-action for {audience}
- 3-5 relevant hashtags on the very last line, space-separated, nothing else on that line
- Professional but conversational tone
- No bullet points, no numbered lists, no emojis

--- IMAGE PROMPT REQUIREMENTS ---
Write a single-line SDXL prompt for a photorealistic image that visually
represents the core message of the post.

The prompt must follow this structure:
  [shot type and subject], [specific scene details], [setting and environment],
  [lighting], [camera/lens details if relevant], photorealistic, cinematic, 8k

Examples of good structure:
  wide shot of a data centre server room at night, blue LED lighting on rows of servers,
  condensation on cold metal racks, a lone engineer checking a screen in the background,
  dramatic atmospheric perspective, photorealistic, cinematic, 8k

  close-up of two professionals shaking hands across a glass conference table,
  city skyline visible through floor-to-ceiling windows, golden afternoon light,
  shallow depth of field, Sony A7R IV 85mm f/1.4, photorealistic, cinematic, 8k

Rules for the image prompt:
- Describe a real physical scene, not abstract concepts or floating text
- 50-80 words
- End with: photorealistic, cinematic, 8k
- No mention of logos, brand names, or copyrighted imagery

Respond in EXACTLY this format (no extra text):
POST:
[full post text - hook, body paragraphs, then hashtags on the last line]
END_POST
IMAGE_PROMPT: [the complete SDXL prompt on one line]
"""


class GenerateContentNode:
    """
    Generates the Bluesky post caption and the SDXL image prompt.
    Assembles final_post so human_approval can display it directly.
    """

    def __init__(
        self,
        cfg: GroqConfig = groq_cfg,
        wf_cfg: WorkflowConfig = workflow_cfg,
    ) -> None:
        self._llm = ChatGroq(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            api_key=cfg.api_key,
        )
        self._wf_cfg = wf_cfg

    def __call__(self, state: BlueskyAgentState) -> BlueskyAgentState:
        approved_news = state.get("approved_news") or []
        topic         = state["topic"]
        audience      = state.get("target_audience") or "professionals and industry leaders"
        human_edits   = state.get("human_edits")

        # Hook is guaranteed to be in state now since generate_hook runs first
        hook = state.get("hook") or ""
        if not hook:
            logger.warning("GenerateContentNode | hook is empty - generate_hook may have failed")

        news_context = "\n\n".join(
            f"Article {i+1}: {n['title']}\n"
            f"Source  : {n['source']}\n"
            f"Content : {n['snippet']}"
            for i, n in enumerate(approved_news)
        )

        edit_section = ""
        if human_edits:
            edit_section = f"\n\nHuman edit instructions - apply these changes:\n{human_edits}"
            logger.info("GenerateContentNode | applying human edits: %r", human_edits)

        prompt = _CONTENT_PROMPT.format(
            topic=topic,
            audience=audience,
            hook=hook,
            news_context=news_context + edit_section,
        )

        logger.info("GenerateContentNode | invoking LLM | topic=%r", topic)
        response = self._llm.invoke(prompt)
        post_text, hashtags, image_prompt = self._parse_response(response.content)

        final_post = BlueskyPost(
            caption=post_text,
            char_count=len(post_text),
        )

        logger.info(
            "GenerateContentNode | post=%d chars | hashtags=%s | image_prompt=%d chars",
            len(post_text), hashtags, len(image_prompt),
        )

        return {
            **state,
            "post_caption":  post_text,
            "hashtags":      hashtags,
            "image_prompt":  image_prompt,
            "final_post":    final_post,
            "human_edits":   None,
        }

    @staticmethod
    def _parse_response(raw: str) -> Tuple[str, List[str], str]:
        """
        Parse Claude's structured response into post text, hashtags, image_prompt.

        Hashtag extraction is now more robust. The old approach required hashtags
        to be on a perfectly clean line with nothing else. Claude sometimes puts
        them inline or appends other words. The new approach:
          1. Scans every line in the post for tokens starting with #
          2. Collects all found hashtags from across the post
          3. Removes all hashtag tokens from the post body so they don't double up
          4. Returns hashtags as a clean list
        This handles all the ways Claude tends to format them.
        """
        post_text    = ""
        image_prompt = ""
        hashtags:    List[str] = []

        # Step 1: extract the POST block between POST: and END_POST
        if "POST:" in raw and "END_POST" in raw:
            start     = raw.index("POST:") + len("POST:")
            end       = raw.index("END_POST")
            post_text = raw[start:end].strip()

        # Step 2: extract IMAGE_PROMPT - handle possible whitespace variations
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("IMAGE_PROMPT:"):
                image_prompt = stripped.split(":", 1)[1].strip()
                break

        # Step 3: scan all lines in the post for hashtag tokens
        # This handles: dedicated hashtag line, inline hashtags, mixed content lines
        if post_text:
            post_lines = post_text.splitlines()
            for line in post_lines:
                tokens = line.split()
                for token in tokens:
                    # Clean trailing punctuation then check for #
                    clean = token.rstrip(".,;:!?")
                    if clean.startswith("#") and len(clean) > 1:
                        hashtags.append(clean)

            if hashtags:
                # Remove all hashtag tokens from post body
                # so they don't show up twice when we append them back later
                cleaned_lines = []
                for line in post_lines:
                    tokens = line.split()
                    kept = [t for t in tokens if not t.rstrip(".,;:!?").startswith("#")]
                    cleaned_line = " ".join(kept).strip()
                    if cleaned_line:
                        cleaned_lines.append(cleaned_line)
                post_text = "\n".join(cleaned_lines).strip()

            # Deduplicate hashtags while preserving order
            seen = set()
            unique_hashtags = []
            for tag in hashtags:
                if tag.lower() not in seen:
                    seen.add(tag.lower())
                    unique_hashtags.append(tag)
            hashtags = unique_hashtags

        return post_text, hashtags, image_prompt