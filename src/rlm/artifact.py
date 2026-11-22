"""The task's final files, replaced only through revision-checked publication."""

from rlm import broker


async def read(path: str) -> dict:
    """A final artifact's state: "revision" (0 until published), "author", "exists",
    "bytes" and "sha256". Read the file itself from the shared workspace."""
    return await broker.agent_request("artifact.read", path=path)


async def publish(
    path: str,
    candidate_path: str,
    *,
    expected_revision: int,
    request_id: str | None = None,
) -> dict:
    """Copy the complete file at candidate_path over the final artifact path, only if
    it is still at expected_revision; raises artifact_conflict otherwise. The candidate's
    bytes are read once, at publication."""
    return await broker.agent_request(
        "artifact.publish",
        path=path,
        candidate_path=candidate_path,
        expected_revision=expected_revision,
        request_id=request_id,
    )
