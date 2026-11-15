"""Orquestração LangGraph das ferramentas do ClassMate."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .extras import generate_study_material
from .rag import answer_question
from .text import text_content

__all__ = ["AgentState", "build_agent", "extract_topic", "requested_activity", "run_agent", "text_content"]


class AgentState(TypedDict, total=False):
    question: str
    user_id: str
    documents: dict[str, str]
    history: list[dict[str, str]]
    route: str
    answer: str
    sources: list[str]


# Verbos de pedido. Exigir um deles evita que "o que diz o resumo do artigo?"
# ou "diferença entre quiz e prova" sejam tratados como comandos.
_ASK = (
    r"(?:fa[çc]a|faz|fazer|crie|cria|criar|gere|gera|gerar|monte|monta|montar|elabore|elaborar|"
    r"prepare|prepara|preparar|quero|queria|gostaria\s+de|me\s+d[êe]|manda|mande)"
)
_ACTIVITY_PATTERNS = [
    ("Quiz", re.compile(rf"\b{_ASK}\b[^?.!]{{0,40}}?\b(?:quiz|question[áa]rio|quest[õo]es|perguntas\s+de\s+m[úu]ltipla\s+escolha)\b")),
    ("Flashcards", re.compile(rf"\b{_ASK}\b[^?.!]{{0,40}}?\b(?:flash\s?cards?|cart[õo]es(?:\s+de\s+estudo)?)\b")),
    ("Resumo", re.compile(rf"^\s*(?:por\s+favor,?\s*)?(?:resuma|sintetize|resume)\b|\b(?:pode|poderia|consegue)\s+(?:me\s+)?(?:resumir|sintetizar)\b|\b{_ASK}\b[^?.!]{{0,40}}?\b(?:resumo|s[íi]ntese)\b")),
    ("Explicação simples", re.compile(r"\bexpli(?:que|ca|car)\b[^?.!]{0,40}?\b(?:como\s+se\s+eu\s+(?:tivesse|fosse)|de\s+(?:forma|jeito|modo)\s+simples|simplificad)|\bexplica[çc][ãa]o\s+simples\b")),
]


def requested_activity(question: str) -> str | None:
    """Reconhece comandos de estudo explícitos sem gastar outra chamada ao modelo."""
    lowered = question.casefold()
    for task, pattern in _ACTIVITY_PATTERNS:
        if pattern.search(lowered):
            return task
    return None


def extract_topic(question: str) -> str:
    """Extrai o tema de pedidos como "faça um quiz sobre fotossíntese"."""
    match = re.search(r"\b(?:sobre|a\s+respeito\s+de|acerca\s+de)\s+(.+)", question, re.IGNORECASE)
    return match.group(1).strip(" ?.!") if match else ""


def build_agent(model: Any, vector_store: Any = None):
    """Cria grafo com três ferramentas: documentos, atividades e chat geral."""
    def route(state: AgentState) -> AgentState:
        docs = state.get("documents", {})
        if docs and vector_store is not None:
            if len(docs) == 1 and requested_activity(state["question"]):
                return {"route": "study"}
            return {"route": "documents"}
        return {"route": "general"}

    def documents(state: AgentState) -> AgentState:
        result = answer_question(
            state["question"], state["user_id"], vector_store, model,
            history=state.get("history", []),
        )
        return {"answer": result["answer"], "sources": result["sources"]}

    def study(state: AgentState) -> AgentState:
        document_id = next(iter(state["documents"]))
        task = requested_activity(state["question"])
        result = generate_study_material(
            task, state["user_id"], document_id, vector_store, model,
            topic=extract_topic(state["question"]),
        )
        return {"answer": result["content"], "sources": result["sources"]}

    def general(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=(
            "Você é o ClassMate AI, um tutor acadêmico. Responda em português brasileiro "
            "de forma didática, clara e precisa. Não afirme ter consultado documentos."
        ))]
        for item in state.get("history", [])[-12:]:
            messages.append(HumanMessage(content=item["content"]) if item["role"] == "user" else AIMessage(content=item["content"]))
        messages.append(HumanMessage(content=state["question"]))
        return {"answer": text_content(model.invoke(messages).content) or "O modelo não retornou uma resposta textual.", "sources": []}

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("documents", documents)
    graph.add_node("study", study)
    graph.add_node("general", general)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda state: state["route"], {
        "documents": "documents", "study": "study", "general": "general",
    })
    for node in ("documents", "study", "general"):
        graph.add_edge(node, END)
    return graph.compile()


def run_agent(question: str, user_id: str, documents: dict[str, str], history: list[dict[str, str]], model: Any, vector_store: Any = None, *, agent: Any = None) -> AgentState:
    """Executa o agente. Passe ``agent`` já compilado para não recriar o grafo a cada pergunta."""
    if not question.strip() or not user_id:
        raise ValueError("Pergunta e usuário são obrigatórios.")
    agent = agent or build_agent(model, vector_store)
    return agent.invoke({"question": question, "user_id": user_id, "documents": documents, "history": history})
