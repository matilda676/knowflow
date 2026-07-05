from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Optional


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def _api_key() -> str:
    return os.getenv("KNOWFLOW_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def _base_url() -> str:
    return os.getenv("KNOWFLOW_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _model() -> str:
    return os.getenv("KNOWFLOW_LLM_MODEL", DEFAULT_MODEL)


def llm_info() -> dict[str, object]:
    return {
        "provider": _base_url(),
        "model": _model(),
        "ready": bool(_api_key()),
    }


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 700,
    timeout: int = 12,
) -> Optional[str]:
    key = _api_key()
    if not key:
        return None

    payload = {
        "model": _model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{_base_url()}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:500]
        except Exception:
            detail = str(exc)
        print(f"[llm] request failed: HTTP {exc.code} {detail}", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        print(f"[llm] request failed: {exc}", file=sys.stderr)
        return None

    choices = result.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        return None
    content = content.strip()
    return content or None


def generate_note_summary(title: str, body: str, category: str) -> Optional[str]:
    if not body.strip():
        return None

    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你是中文个人知识库的摘要助手。只能基于原文总结，不要添加原文没有的信息。"
                    "输出 1 段 80-140 字中文摘要。"
                    "如果是体验/种草类内容，要区分步骤、事实和作者主观反馈。"
                ),
            },
            {
                "role": "user",
                "content": f"标题：{title}\n分类：{category}\n原文：\n{body[:3500]}",
            },
        ],
        temperature=0.1,
        max_tokens=260,
    )


def generate_answer(query: str, contexts: list[dict[str, Any]]) -> Optional[str]:
    blocks = []
    for idx, item in enumerate(contexts[:8], start=1):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] 标题：{item.get('title') or '未命名笔记'}",
                    f"分类：{item.get('category') or '未分类'}",
                    f"片段：{text[:900]}",
                ]
            )
        )

    if not blocks:
        return None

    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你是 KnowFlow 的中文知识库问答助手。只根据给定资料回答，不要编造资料中没有的价格、地点、功效或结论。"
                    "如果资料不足，要明确说资料不足。回答要直接、具体、像真人助理。"
                    "涉及护肤、健康、功效时要说明这是笔记作者主观体验，不当作医学或功效证明。"
                ),
            },
            {
                "role": "user",
                "content": f"用户问题：{query}\n\n可参考资料：\n\n" + "\n\n".join(blocks),
            },
        ],
        temperature=0.25,
        max_tokens=850,
    )
