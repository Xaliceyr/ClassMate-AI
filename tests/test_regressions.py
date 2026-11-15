"""Testes de regressão para os bugs corrigidos na revisão de código."""

import sqlite3

import pytest
from langchain_core.documents import Document

from studybuddy.agent import extract_topic, requested_activity, run_agent
from studybuddy.evaluation import evaluate_result
from studybuddy.extras import generate_study_material, material_context
from studybuddy.memory import clear_messages, create_memory_graph, load_messages, save_turn
from studybuddy.rag import (
    NO_CONTEXT, answer_question, delete_document, get_vector_store, index_file,
    list_user_documents, retrieval_query,
)


class LocalEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class FixedModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": self.content})()


class OneDocStore:
    def similarity_search(self, question, k, filter):
        self.question = question
        return [Document(page_content="x", metadata={"source": "a.pdf", "page": 1})]


# Bug 1: indexação parcial ficava listada como completa.
def test_failed_indexing_rolls_back_partial_batches(tmp_path):
    store = get_vector_store("u", tmp_path, embeddings=LocalEmbeddings())
    original, calls = store.add_documents, {"n": 0}

    def flaky(docs, ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("500 internal")
        return original(docs, ids=ids)

    store.add_documents = flaky
    chunks = [Document(page_content=f"trecho {i}", metadata={"chunk": i}) for i in range(100)]
    with pytest.raises(RuntimeError):
        index_file("u", "grande.txt", b"abc", chunks, store)
    assert list_user_documents("u", store) == {}
    assert store.get(where={"user_id": "u"})["ids"] == []


def test_incomplete_document_is_not_listed(tmp_path):
    store = get_vector_store("u", tmp_path, embeddings=LocalEmbeddings())
    store.add_documents(
        [Document(page_content="só um", metadata={"user_id": "u", "document_id": "d", "source": "x.txt", "chunk": 0, "total_chunks": 3})],
        ids=["d:0:0"],
    )
    assert list_user_documents("u", store) == {}


def test_delete_document_removes_only_that_file(tmp_path):
    store = get_vector_store("u", tmp_path, embeddings=LocalEmbeddings())
    index_file("u", "a.txt", b"a", [Document(page_content="aaa", metadata={"chunk": 0})], store)
    index_file("u", "b.txt", b"b", [Document(page_content="bbb", metadata={"chunk": 0})], store)
    doc_a = next(key for key, name in list_user_documents("u", store).items() if name == "a.txt")
    delete_document("u", doc_a, store)
    assert list(list_user_documents("u", store).values()) == ["b.txt"]


# Bug 2: as respostas do RAG perdiam quebras de linha e listas.
def test_rag_answer_keeps_markdown():
    result = answer_question("q", "u", OneDocStore(), FixedModel("Pontos:\n\n- item A\n- item B"))
    assert result["answer"] == "Pontos:\n\n- item A\n- item B"


def test_abstention_with_extra_punctuation_has_no_sources():
    result = answer_question("q", "u", OneDocStore(), FixedModel(NO_CONTEXT + "  "))
    assert result == {"answer": NO_CONTEXT, "sources": [], "documents": result["documents"]}
    assert evaluate_result({"route": "documents", "answer": NO_CONTEXT, "sources": []})["passed_checks"]


# Bug 4: perguntas comuns eram desviadas para a ferramenta de estudo.
@pytest.mark.parametrize("question", [
    "O que o autor diz no resumo do artigo?",
    "Qual a diferença entre quiz e prova?",
    "Quantas questões tem a prova?",
    "O capítulo explica a fotossíntese?",
])
def test_ordinary_questions_are_not_activities(question):
    assert requested_activity(question) is None


@pytest.mark.parametrize("question,task", [
    ("Faça um quiz sobre RF", "Quiz"),
    ("crie 10 flashcards", "Flashcards"),
    ("Resuma o material", "Resumo"),
    ("Pode resumir o capítulo 2?", "Resumo"),
    ("Gere um resumo sobre embeddings", "Resumo"),
    ("Explique como se eu tivesse 10 anos", "Explicação simples"),
    ("quero uma explicação simples", "Explicação simples"),
])
def test_explicit_commands_are_activities(question, task):
    assert requested_activity(question) == task


# Bug 6: o tema enviado era a frase inteira ("Faça um quiz").
def test_topic_is_extracted_from_command():
    assert extract_topic("Faça um quiz sobre fotossíntese?") == "fotossíntese"
    assert extract_topic("Faça um quiz") == ""


def test_chat_activity_without_topic_uses_overview():
    class Store:
        def get(self, *, where, include):
            return {"documents": ["RF identifica transmissores."], "metadatas": [{"source": "rf.txt", "chunk": 0}]}

        def similarity_search(self, question, k, filter):
            raise AssertionError("sem tema não deve haver busca por similaridade")

    model = FixedModel("1) Pergunta...")
    result = run_agent("Faça um quiz", "u", {"doc": "rf.txt"}, [], model, Store())
    assert result["route"] == "study"
    assert "Visão geral" in model.messages[1].content


def test_topic_uses_relevant_chunks_inside_selected_document():
    class Store:
        def similarity_search(self, query, k, filter):
            self.args = (query, k, filter)
            return [Document(page_content="Trecho sobre antenas", metadata={"source": "rf.pdf", "page": 9, "chunk": 40})]

        def get(self, **_):
            raise AssertionError("não deve usar amostra quando a busca encontra trechos")

    store = Store()
    context, sources = material_context("u", "doc", store, topic="antenas")
    assert store.args == ("antenas", 12, {"$and": [{"user_id": "u"}, {"document_id": "doc"}]})
    assert "antenas" in context and sources == ["rf.pdf, p. 9"]


# Melhoria: troca de assunto não é contaminada pela pergunta anterior.
def test_retrieval_query_uses_history_only_for_follow_ups():
    history = [{"role": "user", "content": "O que é RF fingerprint?"}]
    assert retrieval_query("Quais são as etapas do pré-processamento do sinal no sistema?", history) == \
        "Quais são as etapas do pré-processamento do sinal no sistema?"
    assert retrieval_query("E como ele funciona?", history) == "O que é RF fingerprint?\nE como ele funciona?"


# Melhoria: o SQLite guardava todos os checkpoints e "Limpar" não apagava de fato.
def test_memory_keeps_only_latest_checkpoint_and_clear_deletes(tmp_path):
    path = tmp_path / "memory.sqlite"
    graph = create_memory_graph(path)
    for i in range(5):
        save_turn(graph, "a", f"P{i}", f"R{i}")
    save_turn(graph, "b", "B", "RB")
    assert len(load_messages(graph, "a")) == 10
    count = lambda thread: sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread,)).fetchone()[0]
    assert count("a") == 1
    clear_messages(graph, "a")
    assert count("a") == 0
    assert load_messages(graph, "a") == []
    assert len(load_messages(graph, "b")) == 2
    save_turn(graph, "a", "Nova", "Resposta")
    assert len(load_messages(create_memory_graph(path), "a")) == 2


def test_study_material_preserves_markdown():
    class Store:
        def get(self, **_):
            return {"documents": ["texto"], "metadatas": [{"source": "a.txt", "chunk": 0}]}

    result = generate_study_material("Resumo", "u", "d", Store(), FixedModel("- A\n- B"))
    assert result["content"] == "- A\n- B"
