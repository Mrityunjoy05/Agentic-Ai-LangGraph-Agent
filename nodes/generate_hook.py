"""
nodes/generate_hook.py

Node 3. Now runs sequentially - after human_review_news, before generate_content.

Changed from parallel to sequential so generate_content always gets the
real hook in state rather than a placeholder. This makes the content node
simpler and the hook more coherent with the final post.

The hook selection prompt is also improved. Previously the model just picked
a number. Now it evaluates all 3 hooks against four specific criteria and
picks the one that wins on the most points. Produces more accurate selection.

Reads  : approved_news, topic, target_audience
Writes : hook
"""

from __future__ import annotations

import logging
from typing import Dict

from langchain_groq import ChatGroq

from config.settings import GroqConfig, groq_cfg
from core.state import BlueskyAgentState

logger = logging.getLogger(__name__)

_HOOK_PROMPT = """\
You are a Bluesky content creator with 50,000+ followers.

Topic: {topic}
Target Audience: {audience}

News Articles (read all of these carefully before writing):
{news_context}

Based ONLY on the news articles above, generate exactly {count} hook options
for a Bluesky post opener. Every hook must reference something specific from
the articles - a real number, a real name, a real finding. Do NOT invent facts.

What makes a good Bluesky hook:
- 1-3 lines, not a single cliffhanger sentence
- Opens with a specific insight or surprising fact from the news
- Reads like a thoughtful professional sharing something worth knowing
- No caps-lock, no sensationalism, no "You won't believe..."
- The reader immediately understands why this matters to them professionally
- No hashtags, no "I" statements

Rules:
1. Under 300 characters per hook.
2. Every hook must be grounded in a specific detail from the articles above.
3. No hashtags in any hook.
4. No first-person stories.

After writing all 3 hooks, evaluate each one against these four criteria:
  A. Specificity  - does it cite a concrete fact, number, or name from the news?
  B. Relevance    - will {audience} immediately see why this matters to them?
  C. Credibility  - does it sound like a knowledgeable professional, not a marketer?
  D. Scroll-stop  - would someone pause mid-scroll to read the rest of the post?

Score each hook 1-4 on each criterion (4 = best). Add up the scores.
The hook with the highest total score is the best one.
If two hooks tie, pick the one with the higher Specificity score.

Respond in EXACTLY this format (no extra text):
HOOK_1: [hook text]
HOOK_2: [hook text]
HOOK_3: [hook text]
SCORES:
HOOK_1: A=[1-4] B=[1-4] C=[1-4] D=[1-4] TOTAL=[sum]
HOOK_2: A=[1-4] B=[1-4] C=[1-4] D=[1-4] TOTAL=[sum]
HOOK_3: A=[1-4] B=[1-4] C=[1-4] D=[1-4] TOTAL=[sum]
BEST: [1, 2, or 3]
REASON: [one sentence - cite the specific news detail that makes this the strongest]
"""


class GenerateHookNode:
    """
    Generates 3 Bluesky hook options, scores each one against four criteria,
    and returns the hook with the highest total score.

    Runs sequentially before generate_content so the content node
    always has the real hook available in state.
    """

    def __init__(self, cfg: GroqConfig = groq_cfg) -> None:
        self._llm = ChatGroq(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            api_key=cfg.api_key,
        )

    def __call__(self, state: BlueskyAgentState) -> BlueskyAgentState:
        approved_news = state.get("approved_news") or []
        topic         = state["topic"]
        audience      = state.get("target_audience") or "professionals and industry leaders"

        news_context = "\n\n".join(
            f"Article {i+1}: {n['title']}\n"
            f"Source  : {n['source']}\n"
            f"Content : {n['snippet']}"
            for i, n in enumerate(approved_news)
        )

        prompt = _HOOK_PROMPT.format(
            topic=topic,
            audience=audience,
            news_context=news_context,
            count=3,
        )

        logger.info("GenerateHookNode | invoking LLM | topic=%r", topic)
        response = self._llm.invoke(prompt)
        hook = self._parse_best_hook(response.content)
        logger.info("GenerateHookNode | hook selected (%d chars): %r", len(hook), hook[:80])

        return {**state, "hook": hook}

    @staticmethod
    def _parse_best_hook(raw: str) -> str:
        """
        Parse the scored response and return the hook text chosen by BEST:.
        Falls back to the hook with the highest TOTAL score if BEST: is missing.
        Final fallback is the first hook found.
        """
        hooks: Dict[str, str] = {}
        totals: Dict[str, int] = {}
        best_num = None

        for line in raw.strip().splitlines():
            line = line.strip()

            # Collect hook texts
            if line.startswith("HOOK_") and ":" in line and "A=" not in line:
                key, _, text = line.partition(":")
                num = key.replace("HOOK_", "").strip()
                hooks[num] = text.strip()

            # Collect scores - line looks like: HOOK_1: A=3 B=4 C=3 D=4 TOTAL=14
            elif line.startswith("HOOK_") and "TOTAL=" in line:
                key = line.split(":")[0].replace("HOOK_", "").strip()
                try:
                    total_part = [p for p in line.split() if p.startswith("TOTAL=")]
                    if total_part:
                        totals[key] = int(total_part[0].split("=")[1])
                except (ValueError, IndexError):
                    pass

            # Explicit best pick
            elif line.startswith("BEST:"):
                best_num = line.split(":", 1)[1].strip()

        if not hooks:
            logger.warning("GenerateHookNode | could not parse any hooks, using raw fallback")
            return raw[:300]

        # Use BEST: if present and valid
        if best_num and best_num in hooks:
            logger.info("GenerateHookNode | using BEST=%s", best_num)
            return hooks[best_num]

        # Fall back to highest total score
        if totals:
            best_by_score = max(totals, key=lambda k: totals[k])
            logger.info(
                "GenerateHookNode | BEST: missing, using highest score: hook %s (total=%d)",
                best_by_score, totals[best_by_score],
            )
            return hooks.get(best_by_score, next(iter(hooks.values())))

        # Last resort - return the first hook
        logger.warning("GenerateHookNode | no scores found, returning first hook")
        return next(iter(hooks.values()))