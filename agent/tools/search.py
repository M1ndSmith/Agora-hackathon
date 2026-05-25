"""
Tavily search tool wrapper for LangChain agents + a compact async helper
for deterministic pipelines.

- get_tavily_tool(): returns a LangChain Tool for use inside ReAct agents
- search_compact(): async function that returns a short synthesized answer
  (uses Tavily's include_answer=advanced) — ideal for non-agent pipelines
  where we want to control token budget tightly.
"""
import logging
import os
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)


def get_tavily_tool(max_results: int = 3):
    """
    Return a configured Tavily search tool for agent use (ReAct loops).

    Tries langchain-tavily (new package) first, falls back to
    langchain-community (legacy) if not installed.
    """
    settings = get_settings()
    os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

    try:
        from langchain_tavily import TavilySearch
        return TavilySearch(
            max_results=max_results,
            description=(
                "Search the web for recent news, facts, and information. "
                "Use this to gather evidence about prediction market events. "
                "Input: a specific search query string."
            ),
        )
    except ImportError:
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(
            max_results=max_results,
            tavily_api_key=settings.tavily_api_key,
            description=(
                "Search the web for recent news, facts, and information. "
                "Use this to gather evidence about prediction market events. "
                "Input: a specific search query string."
            ),
        )


async def search_compact(
    query: str,
    max_results: int = 2,
    max_chars: int = 800,
) -> str:
    """
    Token-efficient Tavily search for deterministic pipelines.

    Returns a small string containing Tavily's synthesized answer plus
    short snippets from the top results. Total length capped to ~800 chars
    by default — roughly 200 tokens.

    Falls back to empty string on any error (search failures should not
    break the pipeline).
    """
    settings = get_settings()
    os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=settings.tavily_api_key)
        resp = await client.search(
            query=query,
            max_results=max_results,
            include_answer="advanced",
            search_depth="basic",
        )
    except ImportError:
        # Fallback: use the LangChain wrapper synchronously in a thread
        import asyncio
        tool = get_tavily_tool(max_results=max_results)
        try:
            resp = await asyncio.to_thread(tool.invoke, query)
        except Exception as e:
            logger.warning(f"Tavily fallback failed for '{query[:40]}': {e}")
            return ""
    except Exception as e:
        logger.warning(f"Tavily search failed for '{query[:40]}': {e}")
        return ""

    parts: list[str] = []

    # Tavily answer (synthesized summary)
    if isinstance(resp, dict):
        answer = resp.get("answer")
        if answer:
            parts.append(f"Answer: {answer.strip()}")

        results = resp.get("results", []) or []
        for r in results[:max_results]:
            title = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()
            if content:
                parts.append(f"- {title}: {content[:300]}")
    elif isinstance(resp, list):
        for r in resp[:max_results]:
            if isinstance(r, dict):
                snippet = (r.get("content") or "")[:300]
                if snippet:
                    parts.append(f"- {snippet}")
    elif isinstance(resp, str):
        parts.append(resp[:max_chars])

    joined = "\n".join(parts)
    return joined[:max_chars]
