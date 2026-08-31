"""llm.py — DeepSeek chat-completions client (OpenAI wire-compatible).

DeepSeek exposes an OpenAI-compatible ``/chat/completions`` endpoint, so we
call it with ``requests`` and no extra SDK. The prompt is grounded in the
Context7 snippets already retrieved.
"""
from __future__ import annotations

import os

import requests

DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _system_prompt(context: str) -> str:
    return (
        "You are the 1bit.MONSTER support assistant. You answer questions about "
        "the 1bit.MONSTER engine (a pure C++23, hardware-agnostic 1-bit LLM "
        "inference engine that runs on NPU (Ryzen AI), GPU (ROCm/Vulkan/CUDA), "
        "and CPU; zero Python at runtime; .1bp and GGUF formats; MIT licensed).\n\n"
        "Answer ONLY from the documentation context below. If the context does "
        "not contain the answer, say so plainly and suggest where the user can "
        "find more info. Be concise, accurate, and friendly. Cite source links "
        "from the context when relevant."
        "\n\n---- DOCUMENTATION CONTEXT ----\n" + context
    )


def chat(
    messages: list[dict],
    api_key: str,
    model: str = DEEPSEEK_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout: int = 60,
) -> str:
    """One DeepSeek chat-completions round-trip; returns the reply text."""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(
        DEEPSEEK_BASE + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def generate(
    question: str,
    context: str,
    api_key: str,
    model: str = DEEPSEEK_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout: int = 60,
) -> str:
    """Return a grounded answer for ``question`` using ``context``."""
    return chat(
        [
            {"role": "system", "content": _system_prompt(context)},
            {"role": "user", "content": question},
        ],
        api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
