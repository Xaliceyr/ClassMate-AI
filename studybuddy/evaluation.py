"""Sinais verificáveis de qualidade das respostas do ClassMate."""

from __future__ import annotations

from typing import Any

from .rag import is_abstention


def evaluate_result(result: dict[str, Any]) -> dict[str, Any]:
    """Verifica metadados e estado da resposta, sem julgar verdade factual."""
    route = result.get("route", "")
    answer = str(result.get("answer", "")).strip()
    sources = result.get("sources") or []
    abstained = bool(answer) and is_abstention(answer)
    findings: list[str] = []
    if not answer:
        findings.append("O modelo não retornou texto.")
    if route in {"documents", "study"} and not abstained and not sources:
        findings.append("Resposta sobre material sem fontes exibidas.")
    if route == "general" and sources:
        findings.append("Chat geral apresentou fontes de documentos indevidamente.")
    return {
        "route": route,
        "abstained": abstained,
        "source_count": len(sources),
        "issues": findings,
        "passed_checks": bool(answer) and not findings,
    }
