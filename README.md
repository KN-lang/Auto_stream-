# AutoStream — Social-to-Lead Agent 🚀

> **ServiceHive** | Conversational AI Agent built with LangGraph + Gemini 1.5 Flash / Groq {Llama 3.3 70B, Mixtral 8x7B}

A production-grade AI agent that qualifies social media leads through natural conversation —
classifying user intent, retrieving live knowledge-base answers via RAG, and capturing
name/email/platform **only once all three are validated**.

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────────┐
│  classify_intent    │  Gemini 1.5 Flash + Pydantic structured output
│  (greeting / inquiry│  → intent + confidence + reasoning
│   / high_intent)    │
└────────┬────────────┘
         │
   ┌─────┼──────────────────┐
   ▼     ▼                  ▼
[RAG] [process_lead]   [generate_response]
   │     │
   │   [capture_lead]  ← only fires when name+email+platform all valid
   │     │
   └─────┴──────────────────►[generate_response]
                                      │
                              [summarize?] → END
```

### Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph 0.2+ |
| LLM | Gemini 1.5 Flash |
| Embeddings | Google `models/embedding-001` |
| Vector Store | ChromaDB (persistent) |
| Session Persistence | LangGraph SqliteSaver |
| API | FastAPI |
| CLI | Rich |

---

## Folder Structure

```
autostream/
├── core/
│   ├── config.py         # Pydantic Settings
│   ├── state.py          # AgentState TypedDict
│   └── graph.py          # 6-node LangGraph workflow
├── services/
│   ├── intent_service.py # LLM intent classifier
│   └── rag_service.py    # ChromaDB RAG pipeline
├── tools/
│   └── lead_capture.py   # mock_lead_capture + platform normaliser
├── data/
│   └── kb/autostream.md  # Knowledge base
├── api/
│   └── app.py            # FastAPI: /chat, /webhook, /health
├── tests/                # pytest unit tests
├── main.py               # Rich CLI runner
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the CLI demo

```bash
python main.py
# Or with a custom session ID:
python main.py --thread-id my-session-001
```

### 4. Run the FastAPI server

```bash
uvicorn api.app:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## API Reference

### `POST /chat`

```json
{
  "message": "I want to sign up for the Pro plan",
  "thread_id": "user-123"
}
```

**Response:**
```json
{
  "reply": "That's awesome! I'd love to get you set up...",
  "intent": "high_intent_lead",
  "intent_confidence": 0.97,
  "intent_reasoning": "User explicitly expressed desire to sign up.",
  "lead_data": { "name": null, "email": null, "platform": null },
  "lead_captured": false,
  "turn_count": 1
}
```

### `POST /webhook` (WhatsApp / Twilio)

```json
{
  "From": "whatsapp:+919876543210",
  "Body": "Hi, how much is the Pro plan?"
}
```

---

## WhatsApp Deployment Architecture

```
User (WhatsApp)
  │
  ▼
WhatsApp Business API / Twilio
  │  HTTP POST to /webhook
  ▼
FastAPI (/webhook)          ← phone number used as thread_id
  │
  ▼
LangGraph Agent             ← loads state from SQLite by thread_id
  │
  ├── ChromaDB (RAG)
  └── Gemini 1.5 Flash
  │
  ▼
Response text
  │
  ▼
Twilio Messages API → WhatsApp reply
```

**Steps to deploy:**
1. Register a WhatsApp Business number via [Twilio](https://www.twilio.com/whatsapp) or Meta directly.
2. Deploy this FastAPI app to a public URL (e.g. Railway, Render, AWS).
3. In Twilio Console → Messaging → WhatsApp → set webhook to `https://your-domain/webhook`.
4. Each user's phone number becomes their `thread_id` — state persists in SQLite across sessions.

---

## Intent Classification

| Message | Intent |
|---|---|
| "Hey there!" | `greeting` |
| "How much is the Pro plan?" | `product_inquiry` |
| "What's your refund policy?" | `product_inquiry` |
| "I want to sign up" | `high_intent_lead` |
| "Let's get started" | `high_intent_lead` |
| "Sign me up for Pro" | `high_intent_lead` |

> Curiosity ≠ intent. "How much does Pro cost?" stays `product_inquiry`.

---

## Lead Capture Flow

```
User: I want to sign up
  → "Great! Could I get your name?"
User: My name is Alice
  → "Thanks Alice! What's your email address?"
User: alice@example.com
  → "Perfect! Which platform do you mainly stream on?"
User: YouTube
  → [mock_lead_capture(name="Alice", email="alice@example.com", platform="YouTube")]
  → "You're all set! 🎉 Want to explore our 4K streaming features?"
```

- Platform aliases normalised: "YT" → "YouTube", "IG" → "Instagram", etc.
- Invalid emails trigger a polite re-ask, not a hard failure.
- `mock_lead_capture` is **never called** until all three fields pass validation.

---

## Knowledge Base

Located at `data/kb/autostream.md`. Covers:

- **Pricing**: Basic ($29/mo), Pro ($79/mo), Annual discounts (20% off)
- **Features**: Multi-platform streaming, 4K, analytics, scheduling, webhooks
- **Policies**: 14-day free trial, no refunds after 7 days, cancel anytime, SOC 2 / GDPR

To update the KB and force a re-index:
```python
from services.rag_service import RAGService
RAGService().rebuild()
```

---

## Session Persistence

Sessions are stored in `data/checkpoints.db` (SQLite) via LangGraph's `SqliteSaver`.
Each session is keyed by `thread_id`. The agent resumes mid-conversation even after
a server restart — critical for WhatsApp where users may reply hours later.

Clear a session:
```bash
curl -X DELETE http://localhost:8000/session/user-123
```
