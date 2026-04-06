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
    verbose: bool = False,
    max_iterations: int = 25,
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
    instruction = f"""You are a careful security reconnaissance assistant. Target entry URL: {target_url.strip()}

Process (adapt order: use judgment, you are not a rigid script):
1) Use fetch_headers_for_audit on the entry URL.
2) Use extract_same_host_links on the entry URL.
3) **Reason over observations:** If links suggest login, admin, or API surfaces, or a query parameter worth testing on a lab, call additional tools.
   - Use probe_sensitive_paths_tool with the site's **origin** only (scheme + host + port, no path), e.g. from `http://host/foo` use `http://host`.
   - Use test_sqli_in_parameter **only** when you have a concrete URL that already has a query param (e.g. `...?id=1`) **and** the user context is an authorized lab; never invent a param name.
   - For **test_sqli_in_parameter** the Action Input must be JSON on one line, e.g. {{"url":"http://host/page?id=1","parameter_name":"id"}}.
4) Final Answer: severities (low/medium/high/critical) for each measured issue, grounded only in tool facts. Note uncertainty. Summarize link discovery and any probes you ran. Short overall risk paragraph.

Rules: only report what tools returned; do not claim checks you did not run."""
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
