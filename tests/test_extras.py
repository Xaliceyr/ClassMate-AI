from studybuddy.extras import generate_study_material, material_context
from studybuddy.rag import NO_CONTEXT


class FakeStore:
    def __init__(self, records):
        self.records = records

    def get(self, *, where, include):
        self.where = where
        return self.records

    def similarity_search(self, query, k, filter):
        self.search = (query, k, filter)
        return []


class FakeModel:
    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": "## Resumo\nA radiofrequência identifica emissores."})()


def test_context_filters_user_and_document_and_keeps_sources():
    store = FakeStore({
        "documents": ["Primeira página", "Segunda página"],
        "metadatas": [
            {"user_id": "a", "document_id": "doc", "source": "a.pdf", "page": 1, "chunk": 0},
            {"user_id": "a", "document_id": "doc", "source": "a.pdf", "page": 2, "chunk": 1},
        ],
    })
    context, sources = material_context("a", "doc", store)
    assert store.where == {"$and": [{"user_id": "a"}, {"document_id": "doc"}]}
    assert "Primeira página" in context and "Segunda página" in context
    assert sources == ["a.pdf, p. 1", "a.pdf, p. 2"]


def test_generated_quiz_uses_only_selected_context():
    store = FakeStore({"documents": ["RF fingerprint identifica transmissores."], "metadatas": [{"source": "rf.txt", "chunk": 0}]})
    model = FakeModel()
    result = generate_study_material("Quiz", "a", "doc", store, model, topic="identificação")
    assert "questões de múltipla escolha" in model.messages[1].content
    assert "identificação" in model.messages[1].content
    assert "RF fingerprint" in model.messages[1].content
    assert result["sources"] == ["rf.txt"]


def test_empty_material_does_not_call_model():
    store = FakeStore({"documents": [], "metadatas": []})
    model = FakeModel()
    assert generate_study_material("Resumo", "a", "missing", store, model) == {"content": NO_CONTEXT, "sources": []}
    assert not hasattr(model, "messages")
