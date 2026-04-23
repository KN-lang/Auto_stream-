"""
core/graph.py
=============
LangGraph workflow for the AutoStream Social-to-Lead agent.

Nodes
-----
1. classify_intent_node   — Classifies latest user message
2. retrieve_rag_node      — Fetches relevant KB chunks (product_inquiry path)
3. process_lead_node      — Extracts & validates lead fields (high_intent path)
4. capture_lead_node      — Calls capture_lead (Sheets/mock) once all fields are valid
5. generate_response_node — Produces the final AI reply
6. summarize_node         — Compresses history when turn_count >= threshold
"""

import re
import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from typing import Optional

from core.config import get_settings
from core.state import AgentState
from services.intent_service import IntentService
from services.rag_service import RAGService
from services.llm_factory import resilient_invoke, make_llm_for
from tools.lead_capture import capture_lead, normalize_platform

logger = logging.getLogger(__name__)

_intent_svc: IntentService | None = None
_rag_svc:    RAGService | None    = None


def _get_intent_svc() -> IntentService:
    global _intent_svc
    if _intent_svc is None:
        _intent_svc = IntentService()
    return _intent_svc


def _get_rag_svc() -> RAGService:
    global _rag_svc
    if _rag_svc is None:
        _rag_svc = RAGService()
    return _rag_svc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def _format_history(messages: list, n: int = 8) -> str:
    """Return the last `n` messages as a readable string for prompt injection."""
    lines = []
    for m in messages[-n:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) or "No history yet."


def _get_last_user_message(messages: list) -> str:
    return next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )


# ── Pydantic schema for LLM-based field extraction ───────────────────────────

class ExtractedLeadFields(BaseModel):
    name:     Optional[str] = None
    email:    Optional[str] = None
    platform: Optional[str] = None


# ── Node 1: Classify Intent ───────────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> dict:
    """
    Classifies the latest user message into one of:
      greeting | product_inquiry | high_intent_lead

    Updates: intent, intent_confidence, intent_reasoning, turn_count
    """
    messages = state["messages"]
    last_msg = _get_last_user_message(messages)
    history  = _format_history(messages[:-1])

    result = _get_intent_svc().classify(message=last_msg, history=history)

    logger.info(
        "[classify_intent] intent=%s confidence=%.2f | %s",
        result.intent, result.confidence, result.reasoning,
    )

    return {
        "intent":            result.intent,
        "intent_confidence": result.confidence,
        "intent_reasoning":  result.reasoning,
        "turn_count":        state.get("turn_count", 0) + 1,
    }


# ── Node 2: Retrieve RAG Context ──────────────────────────────────────────────

def retrieve_rag_node(state: AgentState) -> dict:
    """
    Runs similarity search against the ChromaDB knowledge base and stores
    the top-k chunks in `rag_context` for the response node.

    Updates: rag_context
    """
    query   = _get_last_user_message(state["messages"])
    context = _get_rag_svc().retrieve(query)

    logger.debug("[retrieve_rag] Retrieved %d chars of context.", len(context))

    return {"rag_context": context}


# ── Node 3: Process Lead Data ─────────────────────────────────────────────────

def process_lead_node(state: AgentState) -> dict:
    """
    Uses a structured LLM call to extract name, email, and platform from
    the latest user message.  Only fills fields that are still empty.

    Validates email with regex — flags invalid emails separately so the
    response node can ask the user to re-enter without discarding other data.

    Normalises platform aliases (e.g. "YT" → "YouTube").

    Updates: lead_data, is_qualified
    """
    last_msg  = _get_last_user_message(state["messages"])
    lead_data = dict(state.get("lead_data") or {})

    # Ensure all keys exist
    lead_data.setdefault("name",     None)
    lead_data.setdefault("email",    None)
    lead_data.setdefault("platform", None)

    # Remove stale invalid-email flag so we can re-evaluate
    lead_data.pop("_invalid_email", None)

    # ── LLM extraction via resilient structured output ────────────────────────
    s = get_settings()
    models_to_try = [s.groq_model] + list(s.groq_fallback_models)
    import re as _re, time as _time

    def _is_q(e): return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
    def _is_day(e): return "GenerateRequestsPerDayPerProjectPerModel" in str(e)
    def _wait(e):
        m = _re.search(r"retry in (\d+(?:\.\d+)?)s", str(e))
        return float(m.group(1)) if m else 20.0

    prompt = (
        f"Extract name, email, and social media platform from the message below.\n"
        f"Only extract information that is explicitly stated — do not infer.\n"
        f"Message: \"{last_msg}\""
    )

    extracted = None
    for model in models_to_try:
        llm_ex = make_llm_for(model, temperature=0.1)
        extractor = llm_ex.with_structured_output(ExtractedLeadFields)
        attempt = 0
        while attempt < 2:
            try:
                extracted = extractor.invoke(prompt)
                break
            except Exception as exc:
                if not _is_q(exc): raise
                if _is_day(exc): break
                wait = min(_wait(exc) + 2, 35)
                print(f"\n  ⏳  Rate limit — waiting {wait:.0f}s...\n")
                _time.sleep(wait)
                attempt += 1
        if extracted is not None:
            break

    if extracted is None:
        return {"lead_data": lead_data, "is_qualified": False}


    # ── Update name ───────────────────────────────────────────────────────────
    if extracted.name and not lead_data["name"]:
        lead_data["name"] = extracted.name.strip().title()

    # ── Update email (with validation) ────────────────────────────────────────
    if extracted.email and not lead_data["email"]:
        raw_email = extracted.email.strip()
        if _is_valid_email(raw_email):
            lead_data["email"] = raw_email
        else:
            lead_data["_invalid_email"] = raw_email   # signal to response node

    # ── Update platform (with normalisation) ──────────────────────────────────
    if extracted.platform and not lead_data["platform"]:
        canonical = normalize_platform(extracted.platform)
        if canonical:
            lead_data["platform"] = canonical

    is_qualified = bool(
        lead_data.get("name")
        and lead_data.get("email")
        and lead_data.get("platform")
    )

    logger.info(
        "[process_lead] name=%s email=%s platform=%s qualified=%s",
        lead_data.get("name"),
        lead_data.get("email") or lead_data.get("_invalid_email", "–"),
        lead_data.get("platform"),
        is_qualified,
    )

    return {"lead_data": lead_data, "is_qualified": is_qualified}


# ── Node 4: Capture Lead ──────────────────────────────────────────────────────

def capture_lead_node(state: AgentState) -> dict:
    """
    Calls capture_lead() exactly once (guarded by is_qualified + lead_captured checks).
    Writes to Google Sheets if configured, otherwise falls back to mock.

    Updates: lead_captured, rag_context (stores capture receipt for response)
    """
    lead    = state["lead_data"]
    receipt = capture_lead(
        name=lead["name"],
        email=lead["email"],
        platform=lead["platform"],
    )

    logger.info("[capture_lead] Lead captured — id=%s", receipt["lead_id"])

    return {
        "lead_captured": True,
        "rag_context":   f"Lead successfully captured. Receipt: {receipt}",
    }


# ── Node 5: Generate Response ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are AutoStream's friendly and knowledgeable AI sales assistant, built by ServiceHive.

Your responsibilities:
  • Answer questions about AutoStream features, pricing, and policies using the KB context.
  • Guide high-intent users to share their Name, Email, and Platform — one field at a time.
  • After a lead is captured, offer a helpful next step (demo, feature walkthrough, etc.).
  • Keep replies concise and WhatsApp-friendly (≤ 150 words). Use emojis sparingly.

--- Conversation Summary (prior turns) ---
{summary}

--- Retrieved Knowledge Base Context ---
{rag_context}

--- Agent Reasoning ---
Intent     : {intent}  (confidence {confidence:.0%})
Reasoning  : {reasoning}

--- Lead Collection Status ---
{lead_status}

--- Guidelines ---
  1. NEVER invent pricing or features — use ONLY the KB context above.
  2. For product_inquiry: answer accurately from the KB, then invite the user to ask more.
  3. For high_intent_lead: ask for the NEXT missing field (name → email → platform).
     - If an invalid email was provided, politely ask them to re-enter it.
     - Once all three are collected, a confirmation message will follow.
  4. After lead capture, offer a natural next step — do not abruptly end the conversation.
  5. Be warm, professional, and concise.
"""


def generate_response_node(state: AgentState) -> dict:
    """
    Builds a rich system prompt from all state fields and calls the LLM
    to produce the final conversational response.

    Updates: messages (appends new AIMessage)
    """
    lead_data     = state.get("lead_data") or {}
    lead_captured = state.get("lead_captured", False)

    # ── Build human-readable lead status for the system prompt ────────────────
    if lead_captured:
        lead_status = "✅ Lead captured successfully — no further collection needed."
    else:
        collected = {k: v for k, v in lead_data.items()
                     if v and not k.startswith("_")}
        missing   = [k for k in ("name", "email", "platform")
                     if not lead_data.get(k)]
        invalid   = lead_data.get("_invalid_email")

        parts = [f"Collected so far : {collected or 'nothing yet'}"]
        if missing:
            parts.append(f"Still needed     : {missing}")
        if invalid:
            parts.append(
                f"⚠️  Invalid email '{invalid}' was provided — ask them to re-enter."
            )
        lead_status = "\n".join(parts)

    system_content = _SYSTEM_PROMPT.format(
        summary    = state.get("summary",           "No summary yet."),
        rag_context= state.get("rag_context",       "No KB context retrieved."),
        intent     = state.get("intent",            "unknown"),
        confidence = state.get("intent_confidence", 0.0),
        reasoning  = state.get("intent_reasoning",  ""),
        lead_status= lead_status,
    )

    full_messages = [SystemMessage(content=system_content)] + list(state["messages"])
    response      = resilient_invoke(full_messages)

    logger.debug("[generate_response] Response length: %d chars", len(response.content))

    return {"messages": [AIMessage(content=response.content)]}


# ── Node 6: Summarize ─────────────────────────────────────────────────────────

def summarize_node(state: AgentState) -> dict:
    """
    Condenses the conversation into a short summary to preserve context
    across a sliding-window message prune.

    Updates: summary
    Note: We do NOT prune `messages` here — the sliding window is enforced
    by the system prompt injection (last 8 messages passed to the LLM).
    The summary field captures semantic context for older turns.
    """
    history = _format_history(state["messages"], n=10)
    prior   = state.get("summary", "None")

    prompt = (
        f"You are summarising a customer support conversation for AutoStream.\n"
        f"Write a 3-4 sentence summary covering: the user's interest, any pricing "
        f"questions asked, lead information shared (name/email/platform), and the "
        f"current conversation status.\n\n"
        f"Previous summary: {prior}\n\n"
        f"Recent conversation:\n{history}\n\n"
        f"New summary:"
    )

    result = resilient_invoke(prompt)
    logger.info("[summarize] Summary updated (%d chars).", len(result.content))

    return {"summary": result.content}


# ── Conditional Routers ───────────────────────────────────────────────────────

def route_by_intent(
    state: AgentState,
) -> Literal["retrieve_rag", "process_lead", "generate_response"]:
    intent       = state.get("intent", "greeting")
    lead_data    = state.get("lead_data") or {}
    lead_captured = state.get("lead_captured", False)

    # ── If already mid-collection, always route through process_lead ──────────
    # After the FIRST call to process_lead_node, all three keys (name/email/
    # platform) are initialized to None via setdefault. We check key EXISTENCE
    # (not truthiness) so that even when all values are still None we keep
    # routing through process_lead for every subsequent user message.
    if not lead_captured:
        collection_started = any(k in lead_data for k in ("name", "email", "platform"))
        if collection_started:
            return "process_lead"

    if intent == "product_inquiry":
        return "retrieve_rag"
    if intent == "high_intent_lead":
        return "process_lead"
    return "generate_response"


def route_after_lead(
    state: AgentState,
) -> Literal["capture_lead", "generate_response"]:
    if state.get("is_qualified") and not state.get("lead_captured"):
        return "capture_lead"
    return "generate_response"


def route_after_response(
    state: AgentState,
) -> Literal["summarize", "__end__"]:
    if state.get("turn_count", 0) >= get_settings().summarize_after_turns:
        return "summarize"
    return "__end__"


# ── Graph Factory ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Assembles and compiles the LangGraph StateGraph.

    Parameters
    ----------
    checkpointer : optional LangGraph checkpointer (e.g. SqliteSaver).
                   When provided, state is persisted across sessions by thread_id.

    Returns
    -------
    Compiled LangGraph app ready to .invoke() or .stream().
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("classify_intent",   classify_intent_node)
    builder.add_node("retrieve_rag",      retrieve_rag_node)
    builder.add_node("process_lead",      process_lead_node)
    builder.add_node("capture_lead",      capture_lead_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("summarize",         summarize_node)

    # Entry
    builder.add_edge(START, "classify_intent")

    # After classification → branch on intent
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_rag":      "retrieve_rag",
            "process_lead":      "process_lead",
            "generate_response": "generate_response",
        },
    )

    # RAG → response (no branching needed)
    builder.add_edge("retrieve_rag", "generate_response")

    # Lead processing → capture if qualified, else respond
    builder.add_conditional_edges(
        "process_lead",
        route_after_lead,
        {
            "capture_lead":      "capture_lead",
            "generate_response": "generate_response",
        },
    )

    # Capture → response
    builder.add_edge("capture_lead", "generate_response")

    # After response → optionally summarize
    builder.add_conditional_edges(
        "generate_response",
        route_after_response,
        {
            "summarize": "summarize",
            "__end__":   END,
        },
    )

    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)
