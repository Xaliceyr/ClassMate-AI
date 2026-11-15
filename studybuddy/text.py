"""Utilitários de texto compartilhados pelos módulos do ClassMate."""

from __future__ import annotations

from typing import Any


def text_content(content: Any) -> str:
    """Extrai somente os blocos textuais de uma resposta do Gemini.

    O Gemini pode devolver uma string ou uma lista de blocos (texto, raciocínio,
    chamadas de ferramenta). Apenas os blocos do tipo ``text`` são mantidos.
    A formatação Markdown (quebras de linha, listas) é preservada.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content or "").strip()
