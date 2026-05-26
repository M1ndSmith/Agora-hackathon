"""
Tavily search tool wrapper for LangChain agents + a compact async helper
for deterministic pipelines.

- get_tavily_tool(): returns a LangChain Tool for use inside ReAct agents
- search_compact(): async function that returns a short synthesized answer
  (uses Tavily's include_answer=advanced) — ideal for non-agent pipelines
  where we want to control token budget tightly.
- search_weighted(): credibility-ranked snippets for Tier 2 researcher context
"""
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.tools.credibility import rank_snippets, score_source
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


def _parse_published_at(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        s = str(raw).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


async def search_weighted(
    query: str,
    max_results: int = 2,
    max_chars: int = 800,
) -> Dict[str, Any]:
    """
    Credibility-weighted Tavily search for Tier 2 researcher.

    Returns:
        text: compact string for LLM context
        top_source: best domain (str)
        top_score: float
        snippets: list of ranked snippet dicts
    """
    settings = get_settings()
    os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

    snippets: List[dict] = []

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
        # Match search_compact(): fall back to the installed LangChain wrapper
        # when the standalone `tavily` package is not present.
        import asyncio
        tool = get_tavily_tool(max_results=max_results)
        try:
            resp = await asyncio.to_thread(tool.invoke, query)
        except Exception as e:
            logger.warning(f"search_weighted fallback failed for '{query[:40]}': {e}")
            return {"text": "", "top_source": "", "top_score": 0.0, "snippets": []}
    except Exception as e:
        logger.warning(f"search_weighted failed for '{query[:40]}': {e}")
        return {"text": "", "top_source": "", "top_score": 0.0, "snippets": []}

    parts: List[str] = []
    if isinstance(resp, dict):
        answer = resp.get("answer")
        if answer:
            parts.append(f"Answer: {str(answer).strip()}")

        for r in (resp.get("results") or [])[:max_results]:
            url = (r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()
            pub = _parse_published_at(r.get("published_date"))
            sc = score_source(url, pub)
            snippets.append({
                "url": url,
                "title": title,
                "content": content[:300],
                "score": sc,
                "published_at": pub,
            })
    elif isinstance(resp, list):
        for r in resp[:max_results]:
            if not isinstance(r, dict):
                continue
            url = (r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()
            pub = _parse_published_at(r.get("published_date"))
            sc = score_source(url, pub)
            snippets.append({
                "url": url,
                "title": title,
                "content": content[:300],
                "score": sc,
                "published_at": pub,
            })
    elif isinstance(resp, str):
        parts.append(resp[:max_chars])

    ranked = rank_snippets(snippets)
    for s in ranked[:max_results]:
        title = s.get("title") or "source"
        content = s.get("content") or ""
        sc = s.get("score", 0)
        parts.append(f"- [{sc:.2f}] {title}: {content}")

    text = "\n".join(parts)[:max_chars]
    top = ranked[0] if ranked else {}
    top_url = top.get("url", "")
    top_domain = ""
    if top_url:
        from urllib.parse import urlparse
        host = urlparse(top_url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        top_domain = host

    return {
        "text": text,
        "top_source": top_domain,
        "top_score": float(top.get("score", 0)),
        "snippets": ranked,
    }
