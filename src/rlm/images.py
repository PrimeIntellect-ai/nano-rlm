"""Images shown to the model: kernel display bundles and MCP image blocks."""

from __future__ import annotations

import base64
from pathlib import Path

IMAGE_MIMES = ("image/png", "image/jpeg", "image/webp", "image/gif")
MAX_IMAGES_PER_CALL = 4
MAX_IMAGE_BYTES = 2 * 1024 * 1024


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


def attach(path: str | Path) -> None:
    """Show an image file (PNG, JPEG, WebP or GIF, at most 2 MB) to the model.

    The image arrives after this call's output. At most 4 images per call are kept."""
    from IPython.display import display

    data = Path(path).read_bytes()
    mime = sniff_mime(data)
    if mime is None:
        raise ValueError(f"{path}: not a PNG, JPEG, WebP or GIF image")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"{path}: {len(data) / 1e6:.1f} MB exceeds the {MAX_IMAGE_BYTES // 2**20} MB image limit"
        )
    display({mime: base64.b64encode(data).decode()}, raw=True)


def image_part(image: dict) -> dict:
    """An OpenAI chat ``image_url`` content part for ``{"mime", "data"}``."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image['mime']};base64,{image['data']}"},
    }
