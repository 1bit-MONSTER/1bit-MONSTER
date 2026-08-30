"""context7.py — retrieve grounded documentation snippets from Context7.

Thin client for Context7's REST API ``GET /v2/context`` which returns the
most relevant code + info snippets for a library given a free-text question.
See https://context7.com/docs/api-guide and the OpenAPI spec.
"""
from __future__ import annotations

import os
from typing import Any

import requests

CONTEXT7_BASE = os.getenv("CONTEXT7_BASE_URL", "https://context7.com")
CONTEXT_PATH = "/api/v2/context"


def get_context(
    library_id: str,
    query: str,
    api_key: str | None = None,
    fmt: str = "json",
    fast: str = "false",
    timeout: int = 30,
) -> dict[str, Any]:
    """Return the Context7 context response for a library + question."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"libraryId": library_id, "query": query, "type": fmt, "fast": fast}
    resp = requests.get(
        CONTEXT7_BASE + CONTEXT_PATH,
        headers=headers,
        params=params,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _snippet_lines(data: dict[str, Any], max_chars: int = 12000) -> list[str]:
    parts: list[str] = []
    seen: set[str] = set()

    for snip in data.get("infoSnippets", []):
        content = (snip.get("content") or "").strip()
        if not content:
            continue
        breadcrumb = (snip.get("breadcrumb") or snip.get("pageId") or "doc").strip()
        block = f"### {breadcrumb}\n{content}"
        if block not in seen:
            seen.add(block)
            parts.append(block)

    for snip in data.get("codeSnippets", []):
        code_lines = snip.get("codeList") or []
        code = "\n".join(
            (item.get("code") if isinstance(item, dict) else str(item))
            for item in code_lines
        ).strip()
        if not code:
            code = (snip.get("codeDescription") or "").strip()
        if not code:
            continue
        title = (snip.get("codeTitle") or snip.get("pageTitle") or snip.get("sourceFile") or "code").strip()
        block = f"### Code: {title}\n```\n{code}\n```"
        if block not in seen:
            seen.add(block)
            parts.append(block)

    return parts


def format_context(
    data: dict[str, Any], max_chars: int = 12000
) -> str:
    """Flatten snippets + rules into a prompt-ready text block."""
    parts = _snippet_lines(data, max_chars)

    rules = data.get("rules") or {}
    rule_lines: list[str] = []
    for rule_group in ("global", "libraryOwn", "libraryTeam"):
        for rule in rules.get(rule_group, []):
            if rule.strip():
                rule_lines.append(f"- {rule.strip()}")
    if rule_lines:
        parts.append("## Project rules\n" + "\n".join(rule_lines))

    text = "\n\n".join(parts)
    return text[:max_chars]


def source_links(data: dict[str, Any]) -> list[str]:
    """Collect dedup'd source page URLs from the snippets for citations."""
    links: list[str] = []
    seen: set[str] = set()
    for snip in data.get("infoSnippets", []):
        url = snip.get("pageId") or ""
        if url.startswith("http") and url not in seen:
            seen.add(url)
            links.append(url)
    return links
