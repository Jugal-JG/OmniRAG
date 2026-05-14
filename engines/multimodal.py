"""
Multi-Modal engine — mirrors Multi_Modal.ipynb.
LLM: Groq / Llama-4-scout-17b (vision)
Handles image analysis; returns structured description.
"""

import base64
from pathlib import Path

from groq import Groq

from config import Config
from utils import with_retry


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, mime_type)."""
    suffix = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
    mime = mime_map.get(suffix, "image/png")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return data, mime


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    client = Groq(api_key=Config.GROQ_API_KEY)

    content = [{"type": "text", "text": query}]

    for fname in filenames:
        img_path = upload_dir / fname
        if not img_path.exists():
            continue
        b64, mime = _encode_image(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )

    response = client.chat.completions.create(
        model=Config.GROQ_LLM,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [],
        "thinking_steps": [],
    }
