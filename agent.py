"""ReAct agent with fetch_headers + extract_same_host_links tools."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from tools import RECON_TOOLS

# gets path of project and loads the env var
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# used to print/format text in terminal
console = Console()
# model enum
DEFAULT_MODEL = "gpt-4o-mini"
# system prompt for the agent
SYSTEM_PROMPT = """You are a careful security reconnaissance assistant.

Process (adapt order: use judgment, you are not a rigid script):
1) Use fetch_headers_for_audit on the entry URL.
2) Use extract_same_host_links on the entry URL.
3) Reason over observations: If links suggest login, admin, or API surfaces, or a query parameter worth testing on a lab, call additional tools.
   - Use probe_sensitive_paths_tool with the site's origin only (scheme + host + port, no path), e.g. from `http://host/foo` use `http://host`.
   - Use test_sqli_in_parameter only when you have a concrete URL that already has a query param (e.g. `...?id=1`) and the user context is an authorized lab; never invent a param name.
4) Final answer: severities (low/medium/high/critical) for each measured issue, grounded only in tool facts. Note uncertainty. Summarize link discovery and any probes you ran. Short overall risk paragraph.

Rules: only report what tools returned; do not claim checks you did not run."""


def build_agent(
    *,  # must pass arguments with "keyword-only" syntax
    model: str = DEFAULT_MODEL,
    temperature: float = 0,  # controls randomness of model output
) -> CompiledStateGraph:
    llm = ChatOpenAI(model=model, temperature=temperature)
    # a create_agent run returns full message history in order
    return create_agent(llm, RECON_TOOLS, system_prompt=SYSTEM_PROMPT)


def print_trace_message(msg) -> None:
    """Pretty-print a single message from the agent trace (verbose mode)."""
    # langchain has multiple message types, so we need to verify the message is an LLM output
    if isinstance(msg, AIMessage):
        # printing out tool calls
        if msg.tool_calls:
            for call in msg.tool_calls:
                console.print(
                    f"[bold yellow]Action:[/bold yellow] {call['name']}  "
                    f"[dim]{call.get('args', {})}[/dim]"
                )
        text = (msg.content or "").strip() if isinstance(msg.content, str) else ""
        if text:
            console.print(f"[bold cyan]Thought/Answer:[/bold cyan] {text}")
    elif isinstance(msg, ToolMessage):
        # converting preview to string if needed
        preview = (msg.content or "") if isinstance(msg.content, str) else str(msg.content)
        # if the preview is too long, we truncate it
        if len(preview) > 500:
            preview = preview[:500] + "…"
        console.print(f"[bold green]Observation[/bold green] ({msg.name}): {preview}")


def run_audit(
    target_url: str,
    *,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    max_iterations: int = 25,  # cap ReAct loop length
) -> dict:
    agent = build_agent(model=model)
    # lang chain treats this as instructions by a person
    user_message = HumanMessage(
        content=f"Target entry URL: {target_url.strip()}\n\nRun the recon process and return the final answer."
    )
    # recursion_limit bounds total graph steps (LLM + tool nodes); ~2 per ReAct iteration.
    config = {"recursion_limit": max_iterations * 2 + 1}
    inputs = {"messages": [user_message]}

    # if user wants to see the full trace, we print it out
    if verbose:
        final_state: dict = {"messages": []}
        printed = 0
        for chunk in agent.stream(inputs, config=config, stream_mode="values"):
            final_state = chunk
            messages = chunk.get("messages") or []
            # Print every new message since last chunk; parallel tool calls yield
            # multiple ToolMessages appended in a single step.
            for msg in messages[printed:]:
                print_trace_message(msg)
            printed = len(messages)
        return final_state
    # else just invoke the agent
    return agent.invoke(inputs, config=config)


def clean_model_output(text: str) -> str:
    """Strip accidental markdown fences and extra backticks from Final Answer."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:markdown)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    t = t.replace("```", "").strip()
    return t


def extract_final_answer(result: dict) -> str:
    """Pull the last AI message's text content from a create_agent result."""
    # do reversed iteration to get the most recent messages (most recent AI message is the final answer)
    for msg in reversed(result.get("messages", []) or []):
        if isinstance(msg, AIMessage):
            content = msg.content
            # depending on content type, return 
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p)
    return ""


def print_audit_report(target_url: str, result: dict) -> None:
    """Pretty-print agent result (quiet default: only this + optional errors)."""
    # using console library to customize output
    out = clean_model_output(extract_final_answer(result))
    console.print()
    console.print(Rule(f"[bold bright_cyan]Recon — {target_url.strip()}[/]", style="cyan"))
    if not out:
        console.print("[yellow]No Final Answer in result.[/yellow]")
        return
    body = Markdown(out) if out else ""
    console.print(
        Panel(
            body,
            title="[bold]Summary[/bold]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()


def main() -> None:
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run recon ReAct agent on one URL")
    # requiring user to provide a target URL
    parser.add_argument("url", help="Target http(s) URL")
    # option to choose model
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI chat model name",
    )
    # option to print full trace
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full ReAct trace (Thought / Action / Observation)",
    )
    # parsingn and running the audit
    args = parser.parse_args()
    result = run_audit(args.url, model=args.model, verbose=args.verbose)
    print_audit_report(args.url, result)


if __name__ == "__main__":
    main()
