"""Testes das rotas do agente sem chamadas à API."""

from langchain_core.documents import Document

from studybuddy.agent import requested_activity, run_agent
from studybuddy.evaluation import evaluate_result


class FakeModel:
    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": "Resposta fundamentada."})()


class FakeStore:
    def similarity_search(self, question, k, filter):
        self.filter = filter
        return [Document(page_content="RF fingerprint identifica dispositivos.", metadata={"source": "rf.pdf", "page": 2})]

    def get(self, *, where, include):
        self.where = where
        return {"documents": ["RF fingerprint identifica dispositivos."], "metadatas": [{"source": "rf.pdf", "page": 2, "chunk": 0}]}


def test_general_chat_preserves_history_and_has_no_sources():
    model = FakeModel()
    result = run_agent("explique novamente", "a", {}, [{"role": "user", "content": "O que é RF?"}], model)
    assert result["route"] == "general"
    assert result["sources"] == []
    assert "O que é RF?" in model.messages[1].content


def test_document_question_uses_filtered_search():
    store = FakeStore()
    result = run_agent("O que é RF fingerprint?", "a", {"doc": "rf.pdf"}, [], FakeModel(), store)
    assert result["route"] == "documents"
    assert result["sources"] == ["rf.pdf, p. 2"]
    assert store.filter == {"user_id": "a"}
    assert evaluate_result(result)["passed_checks"]


def test_explicit_quiz_routes_to_study_tool():
    store = FakeStore()
    result = run_agent("Faça um quiz", "a", {"doc": "rf.pdf"}, [], FakeModel(), store)
    assert result["route"] == "study"
    assert store.where == {"$and": [{"user_id": "a"}, {"document_id": "doc"}]}
    assert requested_activity("Crie flashcards") == "Flashcards"


def test_evaluation_flags_missing_sources():
    check = evaluate_result({"route": "documents", "answer": "Resposta sem citação", "sources": []})
    assert not check["passed_checks"]
    assert check["issues"]
