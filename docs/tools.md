# Tools and skills

nano-rlm keeps the model's tool list short and puts everything else in Python. By default the model has one native tool, `ipython`, and reaches shell commands, edits, web pages, MCP servers and installed packages as callables in its kernel. Native tool calling is available for runs that want it, and skills teach workflows on top of either.

| Kind | Selected by | The model uses it as |
|---|---|---|
| [Native tools](#native-tools) | `builtin_tools` in the contract | a tool call |
| [Built-in Python tools](#built-in-python-tools) | `skills` in the contract | `await edit(...)` in a cell |
| [Python tool packages](#python-tool-packages) | installed by the host under `$RLM_HOME/skills` | `await websearch(...)`, or a shell command |
| [Skills](#skills) | any folder with a `SKILL.md` under `$RLM_HOME/skills` | instructions it reads when relevant |
| [MCP tools](#mcp-tools) | `mcpServers` in `session/new` | `await tool(...)` after searching the catalog |

## Native tools

`builtin_tools` selects the model's tools; null selects `["ipython"]`.

| Tool | Does |
|---|---|
| `ipython` | Runs a cell in the agent's persistent kernel, with an optional `timeout` up to 600 seconds |
| `bash` | Runs one `bash -c` command in the working directory and returns stdout, stderr and a nonzero exit code |
| `edit` | Replaces one exact occurrence of a string in a file |
| `fetch` | Retrieves a URL as cleaned text |

Whenever `ipython` is active, the model also gets the native `wait` tool ([Inbox and waiting](python-api.md#inbox-and-waiting)). [Kernel and processes](processes.md) describes timeouts, cancellation and what a command's processes outlive.

## Built-in Python tools

The contract's `skills` list pre-imports these into the kernel by name:

| Name | Call |
|---|---|
| `bash` | `await bash("...")`: the `bash` tool's semantics from a cell |
| `edit` | `await edit(path=..., old_str=..., new_str=...)`: a single-occurrence replacement |
| `fetch` | `await fetch(url=...)`: a URL as cleaned text |
| `search` | `await search(query=...)`: web search through Serper or You.com Eco Search (`search_provider`), returning titles, URLs and snippets |

`search` runs in the supervisor through the capability broker, so `search_api_key` never reaches the kernel or its subprocesses; `network_block` filters its results. The others run in the kernel.

## Python tool packages

A host can supply more tools as Python packages. Before running `install.sh`, it places them under `$RLM_HOME/skills/<name>/`; `install.sh` installs them beside rlm, so each is importable in the kernel and its command is on `$PATH`. The prompt lists pre-imported callables as "Python tools (pre-imported)" and packages with console commands as "Shell-enabled Python tools".

```python
import websearch

help(websearch)  # signature and docstring
results = await websearch(queries=["latest jupyter_client release"])
```

```bash
websearch --queries "latest jupyter_client release"
```

A package looks like this:

```text
<name>/
├── SKILL.md            # optional workflow instructions
├── pyproject.toml
└── src/
    └── <name>/
        ├── __init__.py
        └── <name>.py
```

Its one public entry point is an async `run(...)` with type-annotated parameters and a Google-style docstring (`Args:`, `Returns:`), which drive both the Python call and the CLI. The console script points at rlm's shared entry:

```toml
[project.scripts]
<name> = "rlm.skill:cli"
```

`rlm.skill:cli` reads `sys.argv[0]`, imports the matching module, and builds the CLI from `run`'s signature with [tyro](https://brentyi.github.io/tyro/): `queries: list[str]` becomes `--queries`. It prints the return value; an exception surfaces as a traceback. The directory, import name and console script are all `<name>`, and the distribution is `rlm-skill-<name>`; that prefix and the entry point identify a tool package, with or without a `SKILL.md`. Declare `rlm` among the package's dependencies so the entry point resolves; version conflicts between packages installed side by side are the host's to resolve. [The `say` fixture](../tests/fixtures/skills/say) is a minimal example.

Built-in Python tools and MCP callables exist only in the kernel; tool packages also have their command.

## Skills

A skill is a folder holding a `SKILL.md` with `name` and `description` frontmatter: instructions for a workflow, possibly with scripts, references or a tool package. Every such folder under `$RLM_HOME/skills` gets one line in the prompt's `## Skills` section, `- name: description (file: path)`, and the agent reads the full file when its task matches. Its scripts and tools run through the agent's usual tools. A folder with both a `SKILL.md` and a tool package contributes to both listings. The contract's `skills` field selects built-in Python tools, not `SKILL.md` skills.

## MCP tools

A client can give a session [MCP](https://modelcontextprotocol.io) servers in `session/new`, over streamable HTTP or stdio:

```json
{
  "mcpServers": {
    "remote": {"url": "http://127.0.0.1:8000/mcp"},
    "local": {"command": "/path/to/tool-server", "args": ["--stdio"], "env": {"API_KEY": "..."}}
  }
}
```

Programmatically, pass `mcp_servers={"tools": "http://127.0.0.1:8000/mcp"}` to `RLMEngine`, or `{"url": ..., "headers": {...}}` for a server that needs request headers.

At startup the supervisor connects to each server and indexes its whole catalog, schemas included, without creating anything in the kernel. The model finds tools by keyword and loads only what it uses:

```python
await rlm.tools.servers()                              # names, tool counts, descriptions
page = await rlm.tools.search("calendar events", limit=5, server="tools")
tool = await rlm.tools.get(page["matches"][0]["id"])
print(tool.schema)                                     # the full JSON input schema
help(tool)                                             # Python signature and description
result = await tool(day="monday", title="standup")     # returns text
data = json.loads(result)                              # when the tool returns JSON
```

`search` returns up to 20 `matches` per page (ID, server and description) and a `next_cursor`, which continues the same query and server filter; `None` means there are no more. Search is keyword matching, so try other words when nothing fits. Tool names are not listed in the prompt, which shows the discovery guide only when IPython is active and at least one tool was found.

Each call connects with the configured transport, runs the tool and returns its text; a tool-reported error raises `RuntimeError`. Discovery and calls run in the supervisor through the capability broker: the kernel receives only a tool's name, description, public schema and an opaque capability, never server URLs, headers, commands or environment. A catalog is limited to 64 MiB of metadata, 1 MiB per tool and 4,096 pages per server.

## The kernel's environment

The kernel runs in rlm's own Python (`sys.executable`), in the same environment `install.sh` installs rlm and the tool packages into, so their imports work from a cell. To use a project's own packages, shell out to its interpreter: `!./.venv/bin/python -m pytest`.

The kernel starts from a small platform environment (`PATH`, home, user, shell, locale, temporary directory, certificate and virtual-environment variables) plus the contract's `kernel_env`, with private Jupyter and IPython configuration directories. Nothing else from rlm's environment reaches it, which keeps credentials out of reach of the agent's code. This is not containment: in local execution the kernel shares rlm's user, filesystem, processes and network. [Split execution](split-execution.md) moves the tools out.
