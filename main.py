"""
main.py
=======
Rich CLI runner for local development and demo.

Run with:
    python main.py [--thread-id demo-001]
"""

import argparse
import logging
import sqlite3
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import get_settings
from core.graph import build_graph

logging.basicConfig(level=logging.WARNING)
logger  = logging.getLogger(__name__)
console = Console()
settings = get_settings()


def _intent_color(intent: str) -> str:
    return {
        "greeting":          "cyan",
        "product_inquiry":   "yellow",
        "high_intent_lead":  "green",
    }.get(intent, "white")


def run_cli(thread_id: str) -> None:
    # Initialise graph with persistent SQLite checkpointer
    conn          = sqlite3.connect(settings.sqlite_db_path, check_same_thread=False)
    checkpointer  = SqliteSaver(conn)
    graph         = build_graph(checkpointer=checkpointer)
    config        = {"configurable": {"thread_id": thread_id}}

    console.print(Panel.fit(
        "[bold magenta]🚀  AutoStream — Social-to-Lead Agent[/]\n"
        "[dim]Powered by Groq[Llama 3.3 70B, Mixtral 8x7B] + LangGraph + ChromaDB[/]\n\n"
        "[italic]Type your message and press Enter.\n"
        "Commands: [bold]/quit[/] to exit, [bold]/reset[/] to clear session.[/]",
        border_style="magenta",
    ))
    console.print(f"[dim]Session thread_id: {thread_id}[/]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/]")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            console.print("[dim]Goodbye! 👋[/]")
            break

        if user_input.lower() == "/reset":
            try:
                conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
                )
                conn.commit()
                console.print("[yellow]Session cleared. Starting fresh.[/]\n")
            except Exception:
                console.print("[red]Could not clear session (DB may not exist yet).[/]\n")
            continue

        # ── Invoke agent ──────────────────────────────────────────────────────
        with console.status("[dim]Thinking...[/]", spinner="dots"):
            try:
                state = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
            except Exception as exc:
                console.print(f"[red]Agent error:[/] {exc}\n")
                continue

        # ── Extract reply ──────────────────────────────────────────────────────
        ai_messages = [
            m for m in state["messages"]
            if hasattr(m, "content") and not isinstance(m, HumanMessage)
        ]
        reply = ai_messages[-1].content if ai_messages else "…"

        # ── Display reply ──────────────────────────────────────────────────────
        intent     = state.get("intent",            "unknown")
        confidence = state.get("intent_confidence", 0.0)
        reasoning  = state.get("intent_reasoning",  "")
        turn_count = state.get("turn_count", 0)
        lead_data  = {
            k: v for k, v in (state.get("lead_data") or {}).items()
            if not k.startswith("_")
        }
        captured   = state.get("lead_captured", False)

        color = _intent_color(intent)

        # Agent reply panel
        console.print(
            Panel(
                Markdown(reply),
                title="[bold green]AutoStream Assistant[/]",
                border_style="green",
            )
        )

        # Debug metadata bar
        meta = Text()
        meta.append(f" Turn {turn_count} ", style="bold white on dark_blue")
        meta.append(f" {intent} ", style=f"bold white on {color}")
        meta.append(f" {confidence:.0%} confidence ", style="dim")
        if captured:
            meta.append(" ✅ Lead Captured ", style="bold white on dark_green")
        elif lead_data:
            filled = sum(1 for v in lead_data.values() if v)
            meta.append(f" Lead: {filled}/3 fields ", style="dim yellow")
        console.print(meta)
        console.print(f"  [dim italic]Reasoning: {reasoning}[/]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoStream CLI Agent")
    parser.add_argument(
        "--thread-id", default="cli-demo",
        help="Session thread ID (default: cli-demo)",
    )
    args = parser.parse_args()
    run_cli(thread_id=args.thread_id)


if __name__ == "__main__":
    main()
