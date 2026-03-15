"""
nodes/generate_post_content.py

Node 4. Runs in parallel with generate_viral_hook.py.

Generates the body of the thread - tweets 2 through N - along with
hashtags and the CTA. The hook (tweet 1) comes from the other parallel
node and gets merged in by content_optimization afterwards.

Reads  : approved_news, topic, content_type, target_audience, viral_hook
Writes : post_content, hashtags, cta
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_groq import ChatGroq

from config.settings import AnthropicConfig, WorkflowConfig, anthropic_cfg, workflow_cfg
from core.state import TweetThread, TwitterAgentState

logger = logging.getLogger(__name__)

# -- Prompt template
_THREAD_PROMPT = """\
You are a professional Twitter content strategist.

Topic: {topic}
Content Type: {content_type}
Target Audience: {audience}
Opening Hook (already written): {hook}

News Sources to synthesise:
{news_context}

{"Create a Twitter THREAD of " + str(min_t) + "-" + str(max_t) + " tweets." if is_thread else "Create a single impactful tweet."}

Thread structure (for threads):
  Tweet 2 : The core problem or shift happening right now
  Tweet 3 : Key insight drawn from news source 1
  Tweet 4 : Data, numbers, or benchmark stats
  Tweet 5 : What this means for the reader
  Tweet 6 : Emotional close or challenge to the reader
  Tweet 7+ : Additional insights if warranted

Rules:
- Each tweet max 270 chars (leave 10 chars for numbering like "2/7 ").
- Use simple language - no jargon.
- Back every claim with a source from the news context.
- No hashtags in body tweets (hashtags go in the last tweet only).

Respond in EXACTLY this format (no extra text):
TWEET_2: [content]
TWEET_3: [content]
...
HASHTAGS: #tag1 #tag2 #tag3 #tag4 #tag5
CTA: [compelling call-to-action for the final tweet, max 100 chars]
"""


class GeneratePostContentNode:
    """
    Generates the thread body using Claude. The output is raw - numbering
    and final assembly happen in content_optimization.
    """

    def __init__(
        self,
        cfg: AnthropicConfig = anthropic_cfg,
        wf_cfg: WorkflowConfig = workflow_cfg,
    ) -> None:
        self._llm = ChatGroq(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            groq_api_key=cfg.api_key,
        )
        self._wf_cfg = wf_cfg

    def __call__(self, state: TwitterAgentState) -> TwitterAgentState:
        approved_news = state.get("approved_news") or []
        content_type  = state.get("content_type", "thread")
        topic         = state["topic"]
        audience      = state.get("target_audience") or "tech-savvy professionals"
        hook          = state.get("viral_hook") or "TBD - hook generation running in parallel"

        news_context = "\n".join(
            f"Article {i+1}: {n['title']}\n  Snippet : {n['snippet']}\n  Source  : {n['url']}"
            for i, n in enumerate(approved_news)
        )

        prompt = _THREAD_PROMPT.format(
            topic=topic,
            content_type=content_type,
            audience=audience,
            hook=hook,
            news_context=news_context,
            min_t=self._wf_cfg.min_thread_tweets,
            max_t=self._wf_cfg.max_thread_tweets,
            is_thread=(content_type == "thread"),
        )

        logger.info("GeneratePostContentNode | invoking LLM | topic=%r", topic)
        response = self._llm.invoke(prompt)
        tweets, hashtags, cta = self._parse_response(response.content)

        logger.info(
            "GeneratePostContentNode | generated %d tweets | %d hashtags",
            len(tweets),
            len(hashtags),
        )

        return {
            **state,
            "post_content": tweets,
            "hashtags": hashtags,
            "cta": cta,
        }

    @staticmethod
    def _parse_response(raw: str) -> Tuple[List[TweetThread], List[str], str]:
        """Parse the LLM's structured output into typed objects."""
        tweets:   List[TweetThread] = []
        hashtags: List[str]         = []
        cta:      str               = ""

        for line in raw.strip().splitlines():
            line = line.strip()

            if line.startswith("TWEET_"):
                key, _, text = line.partition(":")
                try:
                    num = int(key.replace("TWEET_", "").strip())
                except ValueError:
                    num = len(tweets) + 2

                content = text.strip()
                tweets.append(
                    TweetThread(
                        tweet_number=num,
                        content=content,
                        character_count=len(content),
                    )
                )

            elif line.startswith("HASHTAGS:"):
                raw_tags = line.split(":", 1)[1].strip()
                hashtags = [t for t in raw_tags.split() if t.startswith("#")]

            elif line.startswith("CTA:"):
                cta = line.split(":", 1)[1].strip()

        return tweets, hashtags, cta