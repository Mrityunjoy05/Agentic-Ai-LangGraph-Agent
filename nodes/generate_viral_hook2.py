"""
nodes/generate_viral_hook.py

Node 3. Runs in parallel with generate_post_content.py.

Asks Claude for 3 hook options and picks the best one. Generating multiple
variants and letting the model choose consistently beats asking for one
directly - you get more creative output and the self-selection step filters
out the weak ones.

Reads  : approved_news, topic, target_audience
Writes : viral_hook
"""

from __future__ import annotations

import logging
from typing import Dict

from langchain_groq import ChatGroq

from config.settings import GroqConfig, groq_cfg
from core.state import TwitterAgentState

logger = logging.getLogger(__name__)

_HOOK_PROMPT = """\
You are a viral Twitter content creator with 500K+ followers.

Topic: {topic}
Target Audience: {audience}

Latest News Context:
{news_context}

Generate exactly {count} viral hook options for a tweet/thread opener.

Rules for each hook:
1. Under 280 characters (mandatory).
2. Creates curiosity, urgency, or shock - but NEVER clickbait.
3. Uses power words and specificity (numbers, names, stats).
4. Delivers on its promise - the thread must back it up.
5. No hashtags in the hook.

Respond in EXACTLY this format (no extra text):
HOOK_1: [hook text]
HOOK_2: [hook text]
HOOK_3: [hook text]
BEST: [1, 2, or 3]
REASON: [one sentence why]
"""


class GenerateViralHookNode:
    """
    Generates hook options via Claude and returns whichever the model
    picks as best. Falls back to the first 280 chars of the raw response
    if parsing fails for some reason.
    """

    def __init__(self, cfg: GroqConfig = groq_cfg) -> None:
        self._llm = ChatGroq(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            api_key=cfg.api_key,
        )

    def __call__(self, state: TwitterAgentState) -> TwitterAgentState:
        approved_news = state.get("approved_news") or []
        topic         = state["topic"]
        audience      = state.get("target_audience") or "tech professionals"

        news_context = "\n".join(
            f"- {n['title']} ({n['source']})" for n in approved_news
        )

        prompt = _HOOK_PROMPT.format(
            topic=topic,
            audience=audience,
            news_context=news_context,
            count=3,
        )

        logger.info("GenerateViralHookNode | invoking LLM | topic=%r", topic)
        response = self._llm.invoke(prompt)
        raw      = response.content

        viral_hook = self._parse_best_hook(raw)
        logger.info("GenerateViralHookNode | hook selected (%d chars)", len(viral_hook))

        return  {"viral_hook": viral_hook}

    @staticmethod
    def _parse_best_hook(raw: str) -> str:
        """Pull the chosen hook text out of the formatted LLM response."""
        hooks: Dict[str, str] = {}
        best_num = "1"

        for line in raw.strip().splitlines():
            line = line.strip()
            if line.startswith("HOOK_"):
                key, _, text = line.partition(":")
                hooks[key.replace("HOOK_", "").strip()] = text.strip()
            elif line.startswith("BEST:"):
                best_num = line.split(":", 1)[1].strip()

        if hooks:
            return hooks.get(best_num, next(iter(hooks.values())))

        # If the model didn't follow the format, just use what we got
        logger.warning("GenerateViralHookNode | could not parse hooks, using raw fallback")
        return raw[:280]