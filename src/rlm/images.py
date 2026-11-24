"""Images shown to the model: kernel display bundles and MCP image blocks."""

from __future__ import annotations

import base64
from pathlib import Path

IMAGE_MIMES = ("image/png", "image/jpeg", "image/webp", "image/gif")
MAX_IMAGE_BYTES_PER_CALL = 10 * 1024 * 1024


class ImageBudgetExceeded(ValueError):
    def __init__(self):
        super().__init__(
            "Images exceed the 10 MiB total per tool call. "
            "Attach fewer images per call or compress/resize them."
        )


def decoded_size(data: str) -> int:
    """Decoded bytes in a whitespace-free, padded base64 image."""
    return len(data) * 3 // 4 - (2 if data.endswith("==") else int(data.endswith("=")))


def sniff_mime(data: bytes) -> str | None:
    """The image MIME type from the file signature, or None for other formats."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def attach_image(path: str | Path, *paths: str | Path) -> None:
    """Show one or more image files together (up to 10 MiB total per tool call)."""
    from IPython.display import display

    remaining = MAX_IMAGE_BYTES_PER_CALL
    images = []
    for name in (path, *paths):
        source = Path(name).resolve()
        with source.open("rb") as stream:
            data = stream.read(remaining + 1)
        if len(data) > remaining:
            raise ImageBudgetExceeded()
        mime = sniff_mime(data)
        if mime is None:
            raise ValueError(f"{name}: not a PNG, JPEG, WebP or GIF image")
        images.append((source, mime, data))
        remaining -= len(data)

    for source, mime, data in images:
        display(
            {mime: base64.b64encode(data).decode()},
            metadata={mime: {"path": str(source)}},
            raw=True,
        )


attach = attach_image


def image_part(image: dict) -> dict:
    """An OpenAI chat ``image_url`` content part for ``{"mime", "data"}``."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image['mime']};base64,{image['data']}"},
    }
