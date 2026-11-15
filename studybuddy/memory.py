"""Histórico persistente de conversas com LangGraph e SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from .config import MEMORY_DB
from .text import text_content


def create_memory_graph(db_path: str | Path = MEMORY_DB):
    """Cria um grafo que grava mensagens por thread_id no SQLite."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    # Sobrescreve com zeros o conteúdo apagado, em vez de só marcá-lo como livre.
    connection.execute("PRAGMA secure_delete = ON")
    checkpointer = SqliteSaver(connection)
    builder = StateGraph(MessagesState)
    builder.add_node("save", lambda state: {})
    builder.add_edge(START, "save")
    builder.add_edge("save", END)
    return builder.compile(checkpointer=checkpointer)


def _thread_id(user_id: str) -> str:
    if not user_id:
        raise ValueError("user_id é obrigatório para a memória.")
    return str(user_id)


def _config(user_id: str) -> dict:
    return {"configurable": {"thread_id": _thread_id(user_id)}}


def _prune_old_checkpoints(graph, user_id: str) -> None:
    """Mantém apenas o checkpoint mais recente da conversa.

    Cada checkpoint guarda a lista inteira de mensagens; sem a limpeza, o arquivo
    cresceria de forma quadrática com o tamanho da conversa.
    """
    saver = graph.checkpointer
    if not isinstance(saver, SqliteSaver):
        return
    thread_id = _thread_id(user_id)
    with saver.cursor() as cur:
        for table in ("checkpoints", "writes"):
            cur.execute(
                f"DELETE FROM {table} WHERE thread_id = ? AND checkpoint_ns = '' AND checkpoint_id < "
                "(SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = '')",
                (thread_id, thread_id),
            )


def load_messages(graph, user_id: str) -> list[dict[str, str]]:
    """Lê as mensagens da conversa associada ao usuário."""
    snapshot = graph.get_state(_config(user_id))
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    return [
        {"role": "user" if isinstance(message, HumanMessage) else "assistant", "content": text_content(message.content)}
        for message in messages
        if isinstance(message, (HumanMessage, AIMessage))
    ]


def save_turn(graph, user_id: str, question: str, answer: str) -> None:
    """Salva pergunta e resposta juntas após a resposta ser gerada."""
    graph.invoke(
        {"messages": [HumanMessage(content=question), AIMessage(content=answer)]},
        config=_config(user_id),
    )
    _prune_old_checkpoints(graph, user_id)


def clear_messages(graph, user_id: str) -> None:
    """Apaga fisicamente o histórico desta conversa, mantendo documentos no Chroma."""
    graph.checkpointer.delete_thread(_thread_id(user_id))
