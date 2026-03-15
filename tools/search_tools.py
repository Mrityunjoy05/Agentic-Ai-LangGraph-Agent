"""
tools/search_tools.py

Thin wrapper around Tavily Search. The main reason to wrap it in a class
rather than calling TavilySearchResults directly in the node is testability
- you can pass in a mock SearchTool without any patching gymnastics.

Also keeps Tavily-specific logic (result normalisation, domain extraction)
out of the node, which should stay focused on state management.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_tavily import TavilySearch

from config.settings import TavilyConfig, tavily_cfg
from core.state import NewsItem

logger = logging.getLogger(__name__)


class SearchTool:
    """
    Wraps Tavily and normalises the results into a typed List[NewsItem].

    Pass in a cfg if you need different settings per run. Otherwise it
    picks up the project-wide tavily_cfg singleton automatically.
    """

    def __init__(self, cfg: TavilyConfig = tavily_cfg) -> None:
        self._cfg = cfg
        self._client = TavilySearch(
            max_results= self._cfg.max_results,
            search_depth= self._cfg.search_depth,
            include_answer= self._cfg.include_answer,
            include_raw_content= self._cfg.include_raw_content
        )

    def search(self, topic: str) -> List[NewsItem]:
        """
        Search for recent news on the given topic.

        Raises on non-recoverable Tavily errors so the calling node can
        set workflow_status = 'error' and let error_handler deal with it.
        Returns an empty list only if Tavily returns no results (not an error).
        """
        query = f"latest news about {topic} 2025 2026"
        logger.info("SearchTool.search | query=%r", query)

        raw_results = self._client.invoke(query)

        news_items: List[NewsItem] = []
        for r in raw_results.get('results', []):
            item = NewsItem(
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:500],
                source=self._extract_source(r.get("url", "")),
                published_at=r.get("published_date", "recent"),
            )
            news_items.append(item)

        logger.info("SearchTool.search | fetched %d articles", len(news_items))
        return news_items

    @staticmethod
    def _extract_source(url: str) -> str:
        """Pull the domain out of a URL. Returns the raw URL if parsing fails."""
        try:
            parts = url.split("/")
            return parts[2] if len(parts) > 2 else url
        except Exception:
            return url