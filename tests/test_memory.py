"""Testes da memória persistente do ClassMate."""

from studybuddy.memory import clear_messages, create_memory_graph, load_messages, save_turn


def test_history_survives_new_graph_and_is_isolated(tmp_path):
    path = tmp_path / "memory.sqlite"
    first = create_memory_graph(path)
    save_turn(first, "aluno-a", "O que é RF?", "Radiofrequência.")

    reopened = create_memory_graph(path)
    assert load_messages(reopened, "aluno-a") == [
        {"role": "user", "content": "O que é RF?"},
        {"role": "assistant", "content": "Radiofrequência."},
    ]
    assert load_messages(reopened, "aluno-b") == []


def test_clear_removes_only_selected_conversation(tmp_path):
    graph = create_memory_graph(tmp_path / "memory.sqlite")
    save_turn(graph, "aluno-a", "A", "Resposta A")
    save_turn(graph, "aluno-b", "B", "Resposta B")
    clear_messages(graph, "aluno-a")
    assert load_messages(graph, "aluno-a") == []
    assert len(load_messages(graph, "aluno-b")) == 2
