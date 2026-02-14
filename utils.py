from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_path(path: Path) -> None:
    """
    Ensure directory exists (mkdir -p).
    """
    path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    """
    Lightweight logging helper for CLI.
    """
    print(f"[repo2slides] {message}")


@dataclass
class LLMConfig:
    """
    Simple configuration holder for LLM settings.
    """

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv("OPENAI_API_KEY") or ""
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set and no --api-key provided. "
                "Either export OPENAI_API_KEY or pass --api-key."
            )
        return key


class LLMClient:
    """
    Tiny wrapper around OpenAI Chat Completions API.

    This wrapper is intentionally minimal; users can replace it with their
    own provider implementation if needed.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        # Lazy import to avoid hard dependency if user only wants heuristic mode.
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError(
                "openai package is not installed. Install it via 'pip install openai' "
                "or run repo2slides with --no-llm."
            ) from e

        api_key = self.config.resolved_api_key()
        self._client = OpenAI(api_key=api_key)

    def chat(self, *, system_prompt: str, user_prompt: str) -> str:
        """
        Call the chat completion API and return the response text.
        """
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        # new OpenAI SDK: choices[0].message.content is str | list
        content = response.choices[0].message.content
        if isinstance(content, list):
            # concatenate segments if the SDK returns a list of parts
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""


def safe_read_text(path: Path) -> str:
    """
    Read file content as UTF-8, return empty string on failure.
    """
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def shorten(text: str, max_chars: int = 2000) -> str:
    """
    Shorten long text to limit tokens sent to the LLM, preserving start & end.
    """
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


def to_markdown_bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def dict_get_first(d: Dict[str, Any], keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].strip():
            return d[k]
    return default


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def normalize_for_similarity(text: str) -> str:
    text = normalize_newlines(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_similarity(a: str, b: str) -> float:
    """
    Similarity score in [0, 1] using a blend of token Jaccard and sequence ratio.
    """
    a_n = normalize_for_similarity(a)
    b_n = normalize_for_similarity(b)
    if not a_n or not b_n:
        return 0.0

    seq = SequenceMatcher(None, a_n, b_n).ratio()

    a_tokens = set(_TOKEN_RE.findall(a_n))
    b_tokens = set(_TOKEN_RE.findall(b_n))
    if not a_tokens or not b_tokens:
        jac = 0.0
    else:
        jac = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))

    # Weighted blend: sequence ratio catches near-duplicates; jaccard catches paraphrases a bit.
    return 0.65 * seq + 0.35 * jac

