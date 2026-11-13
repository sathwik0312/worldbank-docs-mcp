"""Agent loop: an OpenAI-compatible LLM (Ollama by default, Groq via env vars) calling the World Bank MCP server.

Usage:
    python client/agent.py "What document types does the World Bank publish most frequently?"
    python client/agent.py            # interactive; blank line or Ctrl-D exits

Env (defaults target local Ollama):
    LLM_BASE_URL=http://localhost:11434/v1   LLM_MODEL=wb-qwen:latest   LLM_API_KEY=ollama
    Groq: LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_MODEL=llama-3.1-8b-instant LLM_API_KEY=<key>

Tool calls and result sizes are logged to stderr; only the final answer goes to stdout.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

SERVER = Path(__file__).resolve().parent.parent / "server" / "wb_docs_mcp.py"
MAX_ITERATIONS = 8
MAX_RESULT_CHARS = 12000  # num_ctx 16384 tokens: room for ~2 full results plus tool schemas

SYSTEM_PROMPT = """You answer questions about World Bank documents using the provided tools.
- Always call a tool before answering.
- Filter values are case-sensitive: when unsure of an exact country, topic, region or type, call get_facets first.
- Cite each document's title, date and pdf link.
- Never invent documents; only mention documents that appear in tool results.
- If a tool result contains a "message" hint, follow it and call again.
- Never ask the user a clarifying question. If a tool returns no results or a hint, follow the hint and try again before answering.
- When the user asks to list all values, list every value the tool returned."""
NUDGE = "Use the tools to find real documents before answering."
FINALIZE = "Stop calling tools. Give your final answer now using only the tool results above."


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


@asynccontextmanager
async def connect():
    """Start the MCP server as a stdio subprocess (same venv) and yield an initialized session."""
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)],
                                   env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"})  # quiet server stderr
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def to_openai_tools(mcp_tools) -> list[dict]:
    return [
        {"type": "function",
         "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema}}
        for t in mcp_tools
    ]


def result_text(result) -> str:
    """Join the text parts of an MCP tool result."""
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


def parse_args(raw) -> dict:
    """Small models send arguments as a JSON string or a dict; raise ValueError on anything else."""
    if isinstance(raw, dict):
        return raw
    args = json.loads(raw or "{}")
    if not isinstance(args, dict):
        raise ValueError(f"arguments must be a JSON object, got {type(args).__name__}")
    return args


async def run_tool(session: ClientSession, name: str, raw_args, tool_names: list[str]) -> str:
    """Execute one tool call. Always returns text for the model; never raises."""
    if name not in tool_names:
        log(f"-> {name} (unknown tool)")
        return f"Tool error: unknown tool {name!r}. Valid tools: {', '.join(tool_names)}."
    try:
        args = parse_args(raw_args)
    except ValueError as e:  # json.JSONDecodeError is a ValueError
        log(f"-> {name} (unparseable args: {raw_args!r})")
        return f"Tool error: could not parse arguments as JSON ({e}). Send a JSON object and call again."

    log(f"-> {name} {json.dumps(args, ensure_ascii=False)}")
    try:
        result = await session.call_tool(name, args)
        text = result_text(result)
        if result.is_error:
            text = f"Tool error: {text}"
    except Exception as e:  # transport/protocol failure: report it, keep the loop alive
        text = f"Tool error: {type(e).__name__}: {e}"

    size = len(text)
    text = fit(text)
    cut = f", trimmed to {len(text)}" if len(text) != size else ""
    log(f"<- {size} chars{cut}{' (error)' if text.startswith('Tool error') else ''}")
    return text


def fit(text: str) -> str:
    """Keep a result under MAX_RESULT_CHARS without ever handing the model broken JSON.

    Document pages lose whole documents from the end, with a note saying how to page on.
    Anything else that is too long is replaced by a short message.
    """
    if len(text) <= MAX_RESULT_CHARS:
        return text
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("documents"), list):
        docs, shown_before = data["documents"], len(data["documents"])
        offset, prior_note = data.get("offset", 0), data.get("note")
        while docs:
            docs.pop()
            n = len(docs)
            more = f"Showing {n} of {shown_before} documents; call again with offset={offset + n} for more."
            data.update(returned=n, next_offset=offset + n, note=f"{prior_note} {more}" if prior_note else more)
            out = json.dumps(data, ensure_ascii=False)
            if len(out) <= MAX_RESULT_CHARS:
                return out
    # ponytail: non-page results this big have not been seen (get_document peaked ~9.4k chars in 30 samples)
    return json.dumps({"message": f"Result too large to show ({len(text)} chars). Ask for less, e.g. fewer rows."})


async def ask(session: ClientSession, llm: AsyncOpenAI, model: str, tools: list[dict], question: str) -> str:
    tool_names = [t["function"]["name"] for t in tools]
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]

    for turn in range(MAX_ITERATIONS):
        reply = (await llm.chat.completions.create(model=model, messages=messages, tools=tools)).choices[0].message
        messages.append(reply.model_dump(exclude_none=True))
        if not reply.tool_calls:
            if turn == 0:  # answered without looking anything up: nudge once
                log("(no tool call on first turn; nudging)")
                messages.append({"role": "user", "content": NUDGE})
                continue
            return reply.content or ""
        for call in reply.tool_calls:
            content = await run_tool(session, call.function.name, call.function.arguments, tool_names)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    log(f"(hit {MAX_ITERATIONS} iterations; asking for a final answer without tools)")
    messages.append({"role": "user", "content": FINALIZE})
    reply = (await llm.chat.completions.create(model=model, messages=messages)).choices[0].message
    return reply.content or ""


def questions():
    from_argv = " ".join(sys.argv[1:]).strip()
    if from_argv:
        yield from_argv
        return
    while True:
        try:
            q = input("Question> ").strip()
        except EOFError:
            return
        if not q:
            return
        yield q


async def main() -> None:
    llm = AsyncOpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                      api_key=os.getenv("LLM_API_KEY", "ollama"))
    model = os.getenv("LLM_MODEL", "wb-qwen:latest")
    async with connect() as session:
        tools = to_openai_tools((await session.list_tools()).tools)
        log(f"model={model} tools={[t['function']['name'] for t in tools]}")
        for q in questions():
            print(await ask(session, llm, model, tools, q), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
