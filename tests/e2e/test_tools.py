"""What a client equips the agent with: MCP servers, `SKILL.md` skills and the native
fetch tool. Each answer exists only behind the feature."""

from __future__ import annotations

import http.server
import sys
import threading
from pathlib import Path

import pytest
from acp.schema import McpServerStdio

from .fixtures.desk_mcp import ECHO_STAMP, ERROR_TOKEN

DESK = McpServerStdio(
    name="desk",
    command=sys.executable,
    args=[str(Path(__file__).parent / "fixtures" / "desk_mcp.py")],
    env=[],
)

CODEWORD = "amber-falcon-482"
CODEWORD_SKILL = f"""---
name: project-codeword
description: Look up the project codeword. Use this whenever you are asked for the project codeword.
---

# Project codeword

The project codeword is `{CODEWORD}`. Reply with it verbatim when asked.
"""

PAGE_STAMP = "pg-41c"


async def test_mcp_tool(run_agent):
    outcome = await run_agent(
        "Find the echo tool of the desk MCP server with rlm.tools, call it with the "
        "message 'hello world', and write exactly what it returns to answer.txt.",
        mcp_servers=[DESK],
    )
    answer = outcome.read("answer.txt")
    assert "hello world" in answer and ECHO_STAMP in answer, outcome.describe()


async def test_mcp_tool_error_reaches_the_model(run_agent):
    outcome = await run_agent(
        "Call the lookup tool of the desk MCP server with the order id A-17, and write "
        "the error message it gives to answer.txt.",
        mcp_servers=[DESK],
    )
    assert ERROR_TOKEN in outcome.read("answer.txt"), outcome.describe()


async def test_skill_instructions(run_agent):
    outcome = await run_agent(
        "What is the project codeword? Look it up with your project-codeword skill and "
        "write it to answer.txt.",
        skill_docs={"project-codeword": CODEWORD_SKILL},
    )
    assert outcome.read("answer.txt") == CODEWORD, outcome.describe()


@pytest.fixture
def notice_url():
    """A page on this machine, so the fetch reaches it from any placement."""

    class Notice(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = f"<html><body><p>The notice stamp is {PAGE_STAMP}.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Notice)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/notice.html"
    server.shutdown()
    thread.join()


async def test_fetch_tool(run_agent, notice_url):
    outcome = await run_agent(
        f"Read {notice_url} with the fetch tool and write the notice stamp it shows to "
        "answer.txt.",
        builtin_tools=["ipython", "fetch"],
    )
    assert outcome.read("answer.txt") == PAGE_STAMP, outcome.describe()
    assert "fetch" in outcome.tool_calls(), outcome.describe()
