"""
services/llm_factory.py
=======================
Resilient LLM factory using Groq (llama-3.3-70b-versatile by default).

Groq provides free, extremely fast inference with generous rate limits.
All LLM calls go through resilient_invoke() which:
  1. Tries the primary model
  2. On per-minute 429: waits the suggested retry delay, then retries once
  3. On quota exhaustion:  falls back to the next model in the fallback list
"""

import re
import time
import logging
from langchain_groq import ChatGroq

from core.config import get_settings

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_retry_seconds(error_str: str) -> float:
    """Extract wait time from rate-limit error messages."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*s(?:econds?)?", error_str)
    return float(m.group(1)) if m else 15.0


def _is_quota_error(exc: Exception) -> bool:
    s = str(exc)
    return ("429" in s or "rate_limit" in s.lower() or "rate limit" in s.lower()) \
           and "decommissioned" not in s.lower()


def _is_decommissioned(exc: Exception) -> bool:
    return "decommissioned" in str(exc).lower() or "model_decommissioned" in str(exc)


def _is_daily_quota(exc: Exception) -> bool:
    s = str(exc).lower()
    return "day" in s and ("limit" in s or "quota" in s)


# ── LLM builders ──────────────────────────────────────────────────────────────

def make_llm(temperature: float = 0.7) -> ChatGroq:
    """Return a ChatGroq client for the primary configured model."""
    s = get_settings()
    return ChatGroq(
        model=s.groq_model,
        api_key=s.groq_api_key,
        temperature=temperature,
        max_retries=0,
    )


def make_llm_for(model: str, temperature: float = 0.7) -> ChatGroq:
    """Return a ChatGroq client for a specific model."""
    s = get_settings()
    return ChatGroq(
        model=model,
        api_key=s.groq_api_key,
        temperature=temperature,
        max_retries=0,
    )


# ── Resilient invoke ──────────────────────────────────────────────────────────

def resilient_invoke(messages, temperature: float = 0.7):
    """
    Call Groq with automatic retry + model fallback.

    Strategy
    --------
    Per-minute rate limit:  wait the suggested seconds, retry same model once.
    Daily quota exhausted:  skip to next fallback model immediately.
    All models exhausted:   raise clear RuntimeError with fix instructions.
    """
    s = get_settings()
    models_to_try = [s.groq_model] + list(s.groq_fallback_models)

    for i, model in enumerate(models_to_try):
        llm = make_llm_for(model, temperature)
        attempt = 0
        while attempt < 2:
            try:
                result = llm.invoke(messages)
                if model != s.groq_model:
                    logger.info("[llm_factory] Using fallback model: %s", model)
                return result

            except Exception as exc:
                if _is_decommissioned(exc):
                    logger.warning("[llm_factory] %s is decommissioned → skip", model)
                    break   # try next model
                if not _is_quota_error(exc):
                    raise   # auth error, bad request etc — bubble up

                if _is_daily_quota(exc):
                    logger.warning("[llm_factory] %s daily quota hit → next model", model)
                    break

                # Per-minute limit — wait and retry
                wait = min(_parse_retry_seconds(str(exc)) + 2, 35)
                logger.warning("[llm_factory] %s rate-limited. Waiting %.0fs...", model, wait)
                print(f"\n  ⏳  Rate limit on {model}. Waiting {wait:.0f}s then retrying...\n")
                time.sleep(wait)
                attempt += 1

    raise RuntimeError(
        "\n\n❌  All Groq models have hit their rate limit.\n"
        "    Free tier limits: ~30 req/min, 14,400 req/day per model.\n\n"
        "    To fix:\n"
        "    1. Wait ~1 minute for the per-minute limit to reset\n"
        "    2. Or visit: https://console.groq.com → upgrade plan\n"
    )
