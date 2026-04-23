from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class LeadData(TypedDict, total=False):
    """Partial lead info collected across turns. All fields start as None."""
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    # Internal flag — set when user provides a malformed email so the
    # response node can ask them to re-enter without losing other data.
    _invalid_email: Optional[str]


class AgentState(TypedDict):
    """
    Central state object threaded through every LangGraph node.

    Fields
    ------
    messages          Full conversation history. `add_messages` reducer
                      appends new messages rather than overwriting.
    summary           Rolling plain-text summary produced by the summarize
                      node; injected into the system prompt so older context
                      is never truly lost after window pruning.
    intent            Last classified intent: one of
                        "greeting" | "product_inquiry" | "high_intent_lead"
    intent_confidence Float 0–1 returned by the intent classifier.
    intent_reasoning  One-sentence explanation from the classifier (aids
                      debugging and makes agent reasoning transparent).
    lead_data         LeadData dict — fields filled in progressively.
    is_qualified      True when name + email + platform are all valid.
    lead_captured     True once mock_lead_capture() has been called.
    turn_count        Incremented on every classify_intent_node call; used
                      to gate summarisation and enforce the 6-turn limit.
    rag_context       Top-k knowledge-base chunks retrieved for the current
                      user message; injected into the system prompt.
    """

    messages:          Annotated[list[BaseMessage], add_messages]
    summary:           str
    intent:            str
    intent_confidence: float
    intent_reasoning:  str
    lead_data:         LeadData
    is_qualified:      bool
    lead_captured:     bool
    turn_count:        int
    rag_context:       str
