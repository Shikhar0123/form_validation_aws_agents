import json
import base64
import boto3
import mimetypes
import re
import os
import unicodedata
from pathlib import Path

client = boto3.client("bedrock-runtime")


# ---------------------------------------------------
# 1. STRICT DOCUMENT NAME SANITIZER (FINAL VERSION)
# ---------------------------------------------------
def sanitize_document_name(input_name: str):
    """
    Returns a filename SAFE for Bedrock Converse document.name.
    Rules:
      Allowed: letters, numbers, spaces, hyphens, parentheses, square brackets
      Not allowed: dot, underscore, emoji, punctuation
      No consecutive spaces
      Extension MUST NOT be included
    """
    # Normalize unicode → ASCII
    name = unicodedata.normalize("NFKD", input_name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Strip extension entirely
    base = os.path.splitext(name)[0]

    # Remove invalid characters
    base = re.sub(r"[^A-Za-z0-9\-\(\)\[\] ]+", " ", base)

    # Collapse multiple spaces
    base = re.sub(r"\s{2,}", " ", base).strip()

    # Fallback
    if not base:
        base = "Document"

    return base  # <─ NO EXTENSION


# ---------------------------------------------------
# 2. BUILD MEDIA PART (Image or Document)
# ---------------------------------------------------
def _build_media_part(
    *, file_path: str = None, media_base64: str = None, media_type: str = None
):
    """Creates AWS Converse-compatible media content block."""

    # ---- Load bytes ----
    if file_path:
        raw = Path(file_path).read_bytes()
    elif media_base64:
        raw = base64.b64decode(media_base64)
    else:
        return None

    # ---- Detect or use provided media_type ----
    if not media_type:
        if not file_path:
            raise ValueError("media_type required when using base64 input")
        media_type = mimetypes.guess_type(file_path)[0]

    if not media_type:
        raise ValueError("Unable to detect media_type for file")

    # ---------- Image ----------
    if media_type.startswith("image/"):
        fmt = media_type.split("/")[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"

        return {
            "image": {
                "format": fmt,
                "source": {"bytes": raw},
            }
        }

    # ---------- Document ----------
    doc_map = {
        "application/pdf": "pdf",
        "text/csv": "csv",
        "application/msword": "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.ms-excel": "xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "text/html": "html",
        "text/plain": "txt",
        "text/markdown": "md",
    }

    fmt = doc_map.get(media_type)
    if not fmt:
        raise ValueError(f"Unsupported document media_type: {media_type}")

    # Filename sanitization
    if file_path:
        raw_name = os.path.basename(file_path)
    else:
        raw_name = f"UploadFile"

    safe_name = sanitize_document_name(raw_name)

    # FINAL valid document block
    return {
        "document": {
            "format": fmt,
            "name": safe_name,  # <── MUST NOT include extension
            "source": {"bytes": raw},
        }
    }


# ---------------------------------------------------
#  ---------------------------------------------------
def bedrock_converse(
    *,
    model_id: str = None,
    inference_profile_arn: str = None,
    system_prompt: str | None = None,
    user_message: str = "",
    file_path: str | None = None,
    media_base64: str | None = None,
    media_type: str | None = None,
    tools: list | None = None,
    tool_choice: dict | None = None,
    inference_config: dict | None = None,
):
    """Fully generic AWS Bedrock Converse wrapper."""

    media_part = _build_media_part(
        file_path=file_path,
        media_base64=media_base64,
        media_type=media_type,
    )

    contents = [{"text": user_message}]
    if media_part:
        contents.append(media_part)

    messages = [{"role": "user", "content": contents}]

    request = {"messages": messages}

    if system_prompt:
        request["system"] = [{"text": system_prompt}]

    if inference_config:
        request["inferenceConfig"] = inference_config

    if tools:
        request["toolConfig"] = {"tools": tools}
        if tool_choice:
            request["toolConfig"]["toolChoice"] = tool_choice

    if inference_profile_arn:
        request["inferenceProfileArn"] = inference_profile_arn
    else:
        request["modelId"] = model_id

    resp = client.converse(**request)

    output = resp.get("output", {})
    msg = output.get("message", {})
    content = msg.get("content", [])

    assistant_text = None
    tool_use = None

    for block in content:
        if "text" in block:
            assistant_text = (assistant_text or "") + block["text"]
        if "toolUse" in block:
            tool_use = block["toolUse"]

    return {
        "assistant_text": assistant_text,
        "tool_use": tool_use,
        "tool_input": tool_use.get("input") if tool_use else None,
        "stopReason": resp.get("stopReason"),
        "usage": resp.get("usage"),
    }
