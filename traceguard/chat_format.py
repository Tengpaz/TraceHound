"""Tokenizer chat-template helpers with a model-agnostic fallback."""

from __future__ import annotations

from typing import Any, Dict, List

from traceguard.prompts import messages_to_plain_prompt


ChatMessage = Dict[str, str]


def messages_to_prompt(messages: List[ChatMessage], *, add_generation_prompt: bool = True) -> str:
    return messages_to_plain_prompt(messages, add_generation_prompt=add_generation_prompt)


def apply_chat_template(
    tokenizer: Any,
    messages: List[ChatMessage],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render messages with tokenizer.chat_template when available.

    Transformers tokenizers expose `apply_chat_template` across modern chat
    models, including InternLM-family checkpoints. Contest-provided tokenizers
    may omit or misconfigure it, so TraceHound keeps a deterministic fallback.
    """

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    return messages_to_prompt(messages, add_generation_prompt=add_generation_prompt)
