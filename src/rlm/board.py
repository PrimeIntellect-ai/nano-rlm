"""The population's shared Markdown board, through the supervisor."""

from rlm import broker


async def read(
    start_line: int = 1, max_lines: int = 200, revision: int | None = None
) -> dict:
    """Read up to max_lines lines from start_line (1-based).

    Returns "text", "start_line", "end_line", "total_lines", "truncated", "revision" and
    "sha256". Pass the revision of an earlier page to page through one consistent
    snapshot; a changed board then raises board_conflict."""
    return await broker.agent_request(
        "board.read", start_line=start_line, max_lines=max_lines, revision=revision
    )


async def search(query: str, limit: int = 20) -> dict:
    """Find board lines containing query: "matches" (line numbers and text),
    "total_matches" and the "revision" searched."""
    return await broker.agent_request("board.search", query=query, limit=limit)


async def append(text: str, *, request_id: str | None = None) -> dict:
    """Append a chunk to the latest board; returns the new "revision".

    Retrying with the same request_id does not append twice."""
    return await broker.agent_request("board.append", text=text, request_id=request_id)


async def edit(
    old: str, new: str, *, expected_revision: int, request_id: str | None = None
) -> dict:
    """Replace the single occurrence of old with new ("" deletes it), only if the board
    is still at expected_revision. Raises board_conflict when it changed since: read it
    again and reconcile."""
    return await broker.agent_request(
        "board.edit",
        old=old,
        new=new,
        expected_revision=expected_revision,
        request_id=request_id,
    )
