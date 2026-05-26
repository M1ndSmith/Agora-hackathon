"""
Multi-LLM ensemble for researcher probability estimates.

Runs only when ensemble_enabled=true. Median-aggregates ai_prob;
downgrades confidence when provider spread exceeds threshold.
"""
import asyncio
import logging
import statistics
from typing import List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm, get_settings, resolve_provider, Settings
from models import EnsembleEstimate, ResearchEstimate

logger = logging.getLogger(__name__)


def _provider_has_key(s: Settings, provider: str) -> bool:
    if provider == "groq":
        return bool(s.groq_api_key and s.groq_api_key != "changeme")
    if provider == "nvidia":
        return bool(s.llm_api_key and s.llm_api_key.startswith("nvapi-"))
    if provider == "openai":
        return bool(s.llm_api_key and s.llm_api_key != "changeme")
    return False


def get_ensemble_llms() -> List[Tuple[str, BaseChatModel]]:
    """
    Return (provider_name, llm) for every configured provider with a valid key.

    Skips the active provider duplicate; always includes at least one LLM.
    """
    s = get_settings()
    active = resolve_provider(s)
    providers = []
    for name in ("groq", "nvidia", "openai"):
        if _provider_has_key(s, name):
            providers.append(name)

    if not providers:
        return [(active, get_llm(streaming=False))]

    seen = set()
    result: List[Tuple[str, BaseChatModel]] = []
    for name in providers:
        if name in seen:
            continue
        seen.add(name)
        try:
            llm = _llm_for_provider(s, name)
            result.append((name, llm))
        except Exception as e:
            logger.warning(f"Ensemble skip provider {name}: {e}")

    if not result:
        return [(active, get_llm(streaming=False))]
    return result


def _llm_for_provider(s: Settings, provider: str) -> BaseChatModel:
    """Build a chat model for a specific provider name."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=s.llm_model,
            api_key=s.groq_api_key,
            temperature=0.3,
            max_tokens=4096,
            streaming=False,
        )
    from langchain_openai import ChatOpenAI
    if provider == "nvidia":
        return ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=s.llm_api_key,
            model=s.llm_model,
            temperature=0.3,
            max_tokens=4096,
            streaming=False,
        )
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        temperature=0.3,
        max_tokens=4096,
        streaming=False,
    )


def aggregate_estimates(
    estimates: List[Tuple[str, ResearchEstimate]],
    disagreement_threshold: float,
) -> EnsembleEstimate:
    """
    Median-aggregate ai_prob across providers.

    If spread > threshold, downgrade_confidence=True.
    """
    if not estimates:
        raise ValueError("aggregate_estimates requires at least one estimate")

    probs = [float(e.ai_prob) for _, e in estimates]
    providers = [name for name, _ in estimates]
    median_prob = statistics.median(probs)
    spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
    downgrade = spread > disagreement_threshold

    # Use estimate from provider closest to median for text fields
    best_idx = min(range(len(probs)), key=lambda i: abs(probs[i] - median_prob))
    anchor = estimates[best_idx][1]

    conf = anchor.confidence or "medium"
    if downgrade:
        conf = "low"

    reasoning = (
        f"Ensemble (N={len(providers)}): {', '.join(providers)} "
        f"-> median {median_prob:.3f}, spread {spread:.3f}"
        + ("; models disagree — confidence downgraded." if downgrade else ".")
    )

    return EnsembleEstimate(
        ai_prob=round(median_prob, 4),
        spread=round(spread, 4),
        providers=providers,
        downgrade_confidence=downgrade,
        reasoning=reasoning,
        key_evidence=anchor.key_evidence or [],
        bull_case=anchor.bull_case or "",
        bear_case=anchor.bear_case or "",
        confidence=conf,
    )


async def run_ensemble_estimate(
    system_prompt: str,
    context: str,
    disagreement_threshold: Optional[float] = None,
) -> EnsembleEstimate:
    """
    Run all ensemble LLMs in parallel and aggregate structured outputs.
    """
    settings = get_settings()
    threshold = disagreement_threshold or settings.ensemble_disagreement_threshold
    llms = get_ensemble_llms()

    async def _one(name: str, llm: BaseChatModel) -> Tuple[str, ResearchEstimate]:
        estimator = llm.with_structured_output(ResearchEstimate)
        result: ResearchEstimate = await estimator.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ])
        return name, result

    results = await asyncio.gather(
        *[_one(name, llm) for name, llm in llms],
        return_exceptions=True,
    )

    ok: List[Tuple[str, ResearchEstimate]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Ensemble provider failed: {r}")
            continue
        ok.append(r)

    if not ok:
        raise RuntimeError("All ensemble providers failed")

    return aggregate_estimates(ok, threshold)
