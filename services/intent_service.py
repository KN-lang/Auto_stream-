from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core.config import get_settings
from services.llm_factory import make_llm_for

# ── Schema ────────────────────────────────────────────────────────────────────

class IntentClassification(BaseModel):
    """Structured output returned by the intent classifier."""

    intent: Literal["greeting", "product_inquiry", "high_intent_lead"] = Field(
        description=(
            "Classified intent of the latest user message. "
            "'greeting' for casual openers or small-talk; "
            "'product_inquiry' for questions about pricing, features, or policies; "
            "'high_intent_lead' for expressions of desire to sign up, start a trial, or purchase."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reasoning: str = Field(
        description="One-sentence explanation of why this intent was chosen."
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_INTENT_PROMPT = """\
You are an intent classifier for AutoStream, a social media streaming SaaS by ServiceHive.

Classify the user's LATEST message into exactly one intent:

  • "greeting"          – Casual hello, thank you, off-topic, or small talk.
  • "product_inquiry"   – Any question about pricing, plans, features, refunds,
                          cancellation, or company policy. Curiosity ≠ intent to buy.
  • "high_intent_lead"  – Explicit desire to sign up, start a trial, purchase, or
                          get started. Examples: "Sign me up", "I want to try Pro",
                          "Let's get started", "How do I subscribe?"

Boundary rules (follow strictly):
  1. "How much does the Pro plan cost?" → product_inquiry  (curiosity)
  2. "I want to try the Pro plan"       → high_intent_lead (action intent)
  3. Never escalate to high_intent_lead on information-seeking messages.
  4. When genuinely ambiguous, prefer product_inquiry over high_intent_lead.

Conversation history (for context only — classify the LATEST message):
{history}

Latest user message: "{message}"

Return valid JSON matching the schema.
"""


# ── Service ───────────────────────────────────────────────────────────────────

class IntentService:
    """
    Classifies user messages using Gemini via the shared resilient LLM factory.
    Falls back to alternative models automatically on rate limits.
    """

    def __init__(self) -> None:
        self._prompt = ChatPromptTemplate.from_template(_INTENT_PROMPT)

    def classify(self, message: str, history: str = "") -> IntentClassification:
        """
        Classify a single user message.

        Uses resilient_invoke() so rate limits are handled transparently —
        waits on per-minute limits, falls back to next model on daily limits.
        """
        from services.llm_factory import make_llm_for, resilient_invoke
        from core.config import get_settings

        formatted = self._prompt.format_messages(
            message=message,
            history=history or "None",
        )

        # Use structured output — need to build a fresh chain against
        # whichever model resilient_invoke selects.
        s = get_settings()
        models_to_try = [s.groq_model] + list(s.groq_fallback_models)

        import re, time, logging
        logger = logging.getLogger(__name__)

        def _is_quota(e): return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
        def _is_daily(e): return "GenerateRequestsPerDayPerProjectPerModel" in str(e)
        def _retry_secs(e):
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e))
            return float(m.group(1)) if m else 20.0

        for model in models_to_try:
            llm = make_llm_for(model, temperature=0.1)
            extractor = llm.with_structured_output(IntentClassification)
            attempt = 0
            while attempt < 2:
                try:
                    return extractor.invoke(formatted)
                except Exception as exc:
                    if not _is_quota(exc):
                        raise
                    if _is_daily(exc):
                        logger.warning("[intent] %s daily quota exhausted", model)
                        break
                    wait = min(_retry_secs(exc) + 2, 35)
                    logger.warning("[intent] %s rate-limited. Waiting %.0fs...", model, wait)
                    print(f"\n  ⏳  Rate limit — waiting {wait:.0f}s...\n")
                    time.sleep(wait)
                    attempt += 1

        raise RuntimeError(
            "All Gemini models hit their daily quota. "
            "Create a new Google Cloud PROJECT (not just a new key) or enable billing."
        )
