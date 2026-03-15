"""
nodes/generate_post_content.py

Node 4. Runs in parallel with generate_viral_hook.py.

Generates the full assembled thread - hook (tweet 1) + body tweets + hashtags
+ CTA, numbered and ready to publish. Since content_optimization is removed,
this node now owns the full assembly: it waits for the hook from state (if
available) or uses a placeholder, builds the complete numbered thread, and
writes it directly to final_draft.

Reads  : approved_news, topic, content_type, target_audience, viral_hook
Writes : post_content, hashtags, cta, final_draft
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_groq import ChatGroq

from config.settings import GroqConfig, WorkflowConfig, groq_cfg, workflow_cfg
from core.state import TweetThread, TwitterAgentState

logger = logging.getLogger(__name__)

_THREAD_PROMPT = """\
You are a professional Twitter content strategist.

Topic: {topic}
Content Type: {content_type}
Target Audience: {audience}
Opening Hook (already written): {hook}

News Sources to synthesise:
{news_context}

{thread_instruction}

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
    Generates the full thread body using Claude, then assembles the complete
    numbered thread (hook + body + CTA + hashtags) and writes it to final_draft
    so human_approval can show it directly without needing content_optimization.
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

    def __call__(self, state: TwitterAgentState) -> TwitterAgentState:
        approved_news = state.get("approved_news") or []
        content_type  = state.get("content_type", "thread")
        topic         = state["topic"]
        audience      = state.get("target_audience") or "tech professionals"

        # viral_hook may or may not be ready yet since both nodes run in parallel.
        # If it's None we use a placeholder -- the hook node writes to state
        # independently and LangGraph merges both outputs before the next node runs.
        hook        = state.get("viral_hook") or "TBD - hook generation running in parallel"
        human_edits = state.get("human_edits")

        news_context = "\n".join(
            f"Article {i+1}: {n['title']}\n  Snippet : {n['snippet']}\n  Source  : {n['url']}"
            for i, n in enumerate(approved_news)
        )

        # If the human left edit instructions from the approval step, append them
        # to the prompt so Claude regenerates with those changes applied.
        edit_section = ""
        if human_edits:
            edit_section = f"\n\nHuman edit instructions to apply:\n{human_edits}"
            logger.info("GeneratePostContentNode | applying human edits: %r", human_edits)

        # Build the thread instruction line in Python first, then pass it
        # as a plain string into the prompt -- .format() can't evaluate
        # Python expressions inside {}, only simple variable substitutions.
        if content_type == "thread":
            thread_instruction = (
                f"Create a Twitter THREAD of "
                f"{self._wf_cfg.min_thread_tweets}-{self._wf_cfg.max_thread_tweets} "
                f"tweets (including the hook as tweet 1)."
            )
        else:
            thread_instruction = "Create a single impactful tweet."

        prompt = _THREAD_PROMPT.format(
            topic=topic,
            content_type=content_type,
            audience=audience,
            hook=hook,
            news_context=news_context + edit_section,
            thread_instruction=thread_instruction,
        )

        logger.info("GeneratePostContentNode | invoking LLM | topic=%r", topic)
        response = self._llm.invoke(prompt)
        body_tweets, hashtags, cta = self._parse_response(response.content)

        logger.info(
            "GeneratePostContentNode | generated %d body tweets | %d hashtags",
            len(body_tweets),
            len(hashtags),
        )

        # Assemble the full thread: hook first, then body, numbered, CTA+hashtags on last tweet
        final_draft = self._assemble_thread(hook, body_tweets, hashtags, cta, content_type)

        return {
            "post_content":  body_tweets,
            "hashtags":      hashtags,
            "cta":           cta,
            "final_draft":   final_draft,
            "human_edits":   None,   # clear after applying so it doesn't re-apply on next run
        }

    def _assemble_thread(
        self,
        hook: str,
        body_tweets: List[TweetThread],
        hashtags: List[str],
        cta: str,
        content_type: str,
    ) -> List[TweetThread]:
        """
        Puts hook + body together, adds 1/N numbering for threads,
        and appends CTA + hashtags to the final tweet.
        """
        hashtag_str = " ".join(hashtags)
        cta_text    = cta or "Follow for more insights!"

        if content_type != "thread":
            # Single tweet - just the hook with hashtags at the end
            content = f"{hook}\n\n{hashtag_str}"
            return [TweetThread(tweet_number=1, content=content, character_count=len(content))]

        # Build ordered list: hook at position 1, then the body tweets
        all_tweets: List[TweetThread] = [
            TweetThread(tweet_number=1, content=hook, character_count=len(hook))
        ]
        for t in body_tweets:
            if t["tweet_number"] != 1:
                all_tweets.append(t)

        total = len(all_tweets)
        assembled: List[TweetThread] = []

        for i, tweet in enumerate(all_tweets, start=1):
            content = f"{i}/{total} {tweet['content']}"

            if i == total:
                content = f"{content}\n\n{cta_text}\n\n{hashtag_str}"

            assembled.append(
                TweetThread(
                    tweet_number=i,
                    content=content,
                    character_count=len(content),
                )
            )

        return assembled

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