"""ReAct agent with fetch_headers + extract_same_host_links tools."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.hub import pull
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from tools import RECON_TOOLS

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

console = Console()
DEFAULT_MODEL = "gpt-4o-mini"


def build_agent_executor(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    verbose: bool = True,
    max_iterations: int = 15,
) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = pull("hwchase17/react-chat")
    agent = create_react_agent(llm, RECON_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=RECON_TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
    )


def run_audit(
    target_url: str,
    *,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> dict:
    """Run header check + link extraction; Final Answer must include judged severities."""
    executor = build_agent_executor(model=model, verbose=verbose)
    instruction = f"""Audit this target URL: {target_url.strip()}

You must:
1) Call fetch_headers_for_audit with that exact URL first.
2) Call extract_same_host_links with the same URL second.
3) In your Final Answer, assign **severity** (exactly one of: low, medium, high, critical) to each distinct finding from the tool results.

Rules:
- Base severities only on **facts** in the tool observations. Do not claim issues you did not measure.
- If context is thin (unknown app sensitivity), prefer **lower** severity and state that limitation.
- For **each** missing security header reported by the header tool, give: header name, severity, one-sentence rationale.
- Summarize same-host links (count; note if zero) and optional severity for attack-surface visibility with brief justification.
- Finish with one short overall risk paragraph."""
    return executor.invoke({"input": instruction, "chat_history": ""})


def _clean_model_output(text: str) -> str:
    """Strip accidental markdown fences and extra backticks from Final Answer."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:markdown)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    t = t.replace("```", "").strip()
    return t


def print_audit_report(target_url: str, result: dict) -> None:
    """Pretty-print agent result (quiet default: only this + optional errors)."""
    out = _clean_model_output(str(result.get("output", "")))
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
    parser = argparse.ArgumentParser(description="Run recon ReAct agent on one URL")
    parser.add_argument("url", help="Target http(s) URL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI chat model name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full ReAct trace (Thought / Action / Observation)",
    )
    args = parser.parse_args()
    result = run_audit(args.url, model=args.model, verbose=args.verbose)
    print_audit_report(args.url, result)


if __name__ == "__main__":
    main()
