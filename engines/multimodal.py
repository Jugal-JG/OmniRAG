"""Multimodal image analysis powered by Gemini 3.1 Flash-Lite."""

from pathlib import Path

from google import genai
from google.genai import types

from answer_format import MATH_FORMAT_INSTRUCTIONS
from config import Config
from utils import with_retry


_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    """Analyze all selected images with the dedicated Gemini vision key."""
    if not Config.GOOGLE_API_KEY2:
        raise RuntimeError(
            "Multimodal analysis requires GOOGLE_API_KEY2 to be configured."
        )

    parts = [
        types.Part.from_text(
            text=(
                f"{query}\n\n"
                "Answer at the level of detail requested by the user. For a request "
                "to explain or summarize in detail, identify the visible elements, "
                "their relationships, and the image's overall meaning. "
                "Do not reduce a detailed request to a one-sentence caption."
                f"{MATH_FORMAT_INSTRUCTIONS}\n"
                "Return only the final answer; do not reveal reasoning or planning."
            )
        )
    ]
    for filename in filenames:
        image_path = upload_dir / filename
        if not image_path.exists():
            continue
        parts.append(
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type=_MIME_TYPES.get(image_path.suffix.lower(), "image/png"),
            )
        )

    client = genai.Client(api_key=Config.GOOGLE_API_KEY2)
    response = client.models.generate_content(
        model=Config.GOOGLE_LLM,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=Config.ANSWER_MAX_TOKENS,
        ),
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty image-analysis response.")

    return {"answer": answer, "sources": [], "thinking_steps": []}
