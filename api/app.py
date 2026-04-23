"""
api/app.py — FastAPI entrypoint for AutoStream agent.
Endpoints: POST /chat, POST /webhook (TwiML), GET /health, DELETE /session/{id}
"""

import logging
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import get_settings
from core.graph import build_graph

logger   = logging.getLogger(__name__)
settings = get_settings()

_graph = None
_checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _checkpointer
    import os
    os.makedirs(os.path.dirname(settings.sqlite_db_path), exist_ok=True)
    conn          = sqlite3.connect(settings.sqlite_db_path, check_same_thread=False)
    _checkpointer = SqliteSaver(conn)
    _graph        = build_graph(checkpointer=_checkpointer)
    logger.info("AutoStream graph initialised.")
    yield
    conn.close()


app = FastAPI(
    title="AutoStream — Social-to-Lead Agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message:   str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply:             str
    intent:            str
    intent_confidence: float
    intent_reasoning:  str
    lead_data:         dict
    lead_captured:     bool
    turn_count:        int


def _invoke_agent(message: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return _graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)


def _last_reply(state: dict) -> str:
    ai = [m for m in state["messages"]
          if hasattr(m, "content") and not isinstance(m, HumanMessage)]
    return ai[-1].content if ai else "Something went wrong — please try again."


def _twiml(message: str) -> Response:
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{message}</Message></Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.get("/health", tags=["Infra"])
def health():
    return {"status": "ok", "service": "AutoStream Agent", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(request: ChatRequest):
    """Standard JSON chat — use for web, Postman, or the eval script."""
    try:
        state = _invoke_agent(request.message, request.thread_id)
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        reply             = _last_reply(state),
        intent            = state.get("intent",            "unknown"),
        intent_confidence = state.get("intent_confidence", 0.0),
        intent_reasoning  = state.get("intent_reasoning",  ""),
        lead_data         = {k: v for k, v in (state.get("lead_data") or {}).items()
                             if not k.startswith("_")},
        lead_captured     = state.get("lead_captured", False),
        turn_count        = state.get("turn_count",    0),
    )


@app.post("/webhook", tags=["WhatsApp"])
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    To:   str = Form(""),
):
    """
    Twilio WhatsApp webhook — parses form-encoded payload, returns TwiML.

    Setup:
    1. Deploy this API to Railway
    2. Twilio Console → WhatsApp Sandbox → Webhook URL:
       https://your-app.up.railway.app/webhook
    """
    import os
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if auth_token:
        try:
            from twilio.request_validator import RequestValidator
            from urllib.parse import parse_qs
            body_bytes = await request.body()
            params = {k: v[0] for k, v in parse_qs(body_bytes.decode()).items()}
            sig = request.headers.get("X-Twilio-Signature", "")
            if not RequestValidator(auth_token).validate(str(request.url), params, sig):
                return _twiml("Unauthorized.")
        except Exception as exc:
            logger.warning("Signature validation error: %s", exc)

    thread_id = From.replace("whatsapp:", "").replace("+", "").strip()
    logger.info("[webhook] From=%s msg=%r", From, Body[:80])

    try:
        state = _invoke_agent(Body, thread_id)
        reply = _last_reply(state)
    except Exception as exc:
        logger.exception("Webhook error: %s", exc)
        reply = "Sorry, I'm having a little trouble right now. Please try again shortly! 🙏"

    return _twiml(reply)


@app.delete("/session/{thread_id}", tags=["Infra"])
def clear_session(thread_id: str):
    """Clear a session from SQLite (useful for testing)."""
    try:
        conn = sqlite3.connect(settings.sqlite_db_path)
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "cleared", "thread_id": thread_id}
