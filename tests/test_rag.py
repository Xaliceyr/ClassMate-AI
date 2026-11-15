from langchain_core.documents import Document

from studybuddy.rag import NO_CONTEXT, answer_question, format_sources, get_vector_store, index_file, list_user_documents, retrieve
from studybuddy import rag


class FakeStore:
    def __init__(self, results=None):
        self.results = results or []
        self.added = []

    def add_documents(self, docs, ids):
        self.added.append((docs, ids))

    def similarity_search(self, question, k, filter):
        self.last_filter = filter
        return self.results[:k]


class FakeModel:
    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": "Resposta baseada no material."})()


def test_index_sets_user_id_and_stable_ids():
    store = FakeStore()
    count = index_file("aluno-1", "a.txt", b"conteudo", [Document(page_content="trecho", metadata={"chunk": 0})], store)
    assert count == 1
    docs, ids = store.added[0]
    assert docs[0].metadata["user_id"] == "aluno-1"
    assert docs[0].metadata["source"] == "a.txt"
    assert ids[0].startswith("aluno-1:")


def test_retrieval_always_filters_by_user():
    store = FakeStore()
    retrieve("pergunta", "aluno-2", store)
    assert store.last_filter == {"user_id": "aluno-2"}


def test_no_results_returns_abstention_without_calling_model():
    model = FakeModel()
    result = answer_question("pergunta", "aluno", FakeStore(), model)
    assert result["answer"] == NO_CONTEXT
    assert result["sources"] == []
    assert not hasattr(model, "messages")


def test_answer_uses_retrieved_context_and_returns_citations():
    docs = [Document(page_content="Fotossíntese converte luz em energia química.", metadata={"source": "bio.pdf", "page": 3, "user_id": "a"})]
    model = FakeModel()
    result = answer_question("O que é fotossíntese?", "a", FakeStore(docs), model)
    assert result["answer"] == "Resposta baseada no material."
    assert result["sources"] == ["bio.pdf, p. 3"]
    assert "Fotossíntese" in model.messages[1].content


def test_sources_are_deduplicated():
    docs = [Document(page_content="a", metadata={"source": "x.pdf", "page": 2}), Document(page_content="b", metadata={"source": "x.pdf", "page": 2})]
    assert format_sources(docs) == ["x.pdf, p. 2"]


def test_index_retries_minute_limit_after_server_delay(monkeypatch):
    class LimitedStore(FakeStore):
        calls = 0

        def add_documents(self, docs, ids):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED: requests per minute; Please retry in 55.2s")
            super().add_documents(docs, ids)

    waits = []
    monkeypatch.setattr(rag.time, "sleep", waits.append)
    store = LimitedStore()
    count = index_file("a", "a.txt", b"x", [Document(page_content="a", metadata={"chunk": 0})], store)
    assert count == 1
    assert store.calls == 2
    assert waits == [57]


def test_documents_survive_reopening_chroma_and_remain_isolated(tmp_path):
    class LocalEmbeddings:
        def embed_documents(self, texts):
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0]

    path = tmp_path / "chroma"
    first = get_vector_store("aluno-a", path, embeddings=LocalEmbeddings())
    index_file("aluno-a", "apostila.txt", b"conteudo", [Document(page_content="conteúdo da apostila", metadata={"chunk": 0})], first)
    reopened = get_vector_store("aluno-a", path, embeddings=LocalEmbeddings())
    assert list_user_documents("aluno-a", reopened)
    assert list(list_user_documents("aluno-a", reopened).values()) == ["apostila.txt"]
    assert list_user_documents("aluno-b", reopened) == {}
