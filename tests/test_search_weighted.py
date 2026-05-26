import asyncio


def test_search_weighted_falls_back_to_langchain_tool(monkeypatch):
    from agent.tools import search

    class FakeTool:
        def invoke(self, query):
            assert "PSG" in query
            return [
                {
                    "title": "Reuters soccer report",
                    "url": "https://www.reuters.com/sports/soccer/psg",
                    "content": "PSG advanced after a strong Champions League run.",
                },
                {
                    "title": "Fan blog",
                    "url": "https://fan-blog.example/psg",
                    "content": "A fan take on PSG odds.",
                },
            ]

    monkeypatch.setattr(search, "get_tavily_tool", lambda max_results=2: FakeTool())

    result = asyncio.run(
        search.search_weighted("Will PSG win the Champions League?", max_results=2)
    )

    assert "Reuters soccer report" in result["text"]
    assert result["top_source"] == "reuters.com"
    assert result["top_score"] > 0
    assert len(result["snippets"]) == 2


def test_search_weighted_fallback_failure_returns_empty(monkeypatch):
    from agent.tools import search

    class BrokenTool:
        def invoke(self, query):
            raise RuntimeError("down")

    monkeypatch.setattr(search, "get_tavily_tool", lambda max_results=2: BrokenTool())

    result = asyncio.run(search.search_weighted("anything", max_results=2))

    assert result == {"text": "", "top_source": "", "top_score": 0.0, "snippets": []}
