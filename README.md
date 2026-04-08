# recon-agent

An **LLM-driven security reconnaissance agent**: you give it a **target URL** (only systems you own or are allowed to test), and it **decides which checks to run** in a loop, using **tools** you implement in Python. It is a learning/demo agent—not a replacement for professional scanners or authorized assessments.

## What the agent is doing

The agent follows a **ReAct-style loop** (reason → act → read the result → repeat):

1. The **model** reads your task (e.g. “audit this URL”) and any **observations** from earlier steps.
2. It emits a **tool call**: a **name** (which function to run) and **arguments** (e.g. the URL).
3. **LangChain** runs your **tool**—ordinary code: `requests.get`, BeautifulSoup, etc.
4. The **return value** of that function is fed back to the model as the **observation**.
5. The loop continues until the model produces a **final answer** (and optionally until you merge in a structured **findings** list from the tools).

So: the **model plans and orchestrates**; the **tools perform** HTTP/HTML work and encode security logic (missing headers, interesting links, basic injection checks against a lab app, etc.).

## How the code is meant to be organized

Once implemented, the repo is structured roughly like this:

- **`tools.py`** — The real “scanner” pieces: functions such as `fetch_headers` and `extract_links`, decorated with `@tool` so the agent can call them. Each tool’s **docstring** tells the model when to use it. Tools can **append** to a shared **findings** list so you get structured rows regardless of the chatty final summary.
- **`agent.py`** — Wires **`ChatOpenAI`** to a **ReAct agent** (e.g. LangChain’s `AgentExecutor` + `create_react_agent` in `langchain-classic`) with the tool list, **system prompt**, and **`verbose`** logging so you can see Thought / Action / Observation in the terminal.
- **`report.py`** — Turns the collected **findings** into **JSON** (and later optional Markdown) for demos and interviews.

## What you should see when you run it

With `verbose` logging enabled, a run should show **multiple tool invocations** (for example headers first, then link extraction), then a short **natural-language summary**. Separately, you can print or save a **JSON array** of findings (`vuln_type`, `endpoint`, `severity`, `evidence`, `recommendation`, etc.) built from what the tools recorded.

## Safety

Use **local lab targets** (e.g. DVWA in Docker) for demonstration. Do not aim this at hosts you are not explicitly permitted to test.
