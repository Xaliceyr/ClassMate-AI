"""Aplicação Streamlit do ClassMate AI com chat e RAG para PDF/TXT."""

import hashlib
import os
import re
from uuid import UUID, uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from studybuddy.agent import build_agent, run_agent
from studybuddy.config import BASE_DIR, chat_model_name
from studybuddy.evaluation import evaluate_result
from studybuddy.extras import TASKS, generate_study_material
from studybuddy.ingestion import load_uploaded_file, split_documents, stable_document_id
from studybuddy.memory import clear_messages, create_memory_graph, load_messages, save_turn
from studybuddy.rag import delete_document, get_vector_store, index_file, is_quota_error, list_user_documents

load_dotenv(BASE_DIR / ".env")
st.set_page_config(page_title="ClassMate AI", page_icon="📚", layout="wide")
st.title("📚 ClassMate AI")
st.caption("Seu tutor pessoal com inteligência artificial")

ROUTE_LABELS = {"documents": "consulta ao material", "study": "atividade de estudo", "general": "conversa geral"}


def md_escape(text: str) -> str:
    """Evita que nomes de arquivo sejam interpretados como Markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~])", r"\\\1", text)


def friendly_error(exc: Exception) -> str:
    if is_quota_error(exc):
        return "Limite temporário da API Gemini atingido. Aguarde um minuto e tente novamente."
    return str(exc)


# O ID no endereço permite reencontrar os vetores após atualizar a página.
# Em implantação com login, substitua por um ID do usuário autenticado.
try:
    user_id = str(UUID(st.query_params["studybuddy_id"]))
except (KeyError, ValueError, TypeError):
    user_id = str(uuid4())
    st.query_params["studybuddy_id"] = user_id

# Se o ID da URL mudar na mesma aba, o estado da sessão anterior não pode vazar.
if st.session_state.get("studybuddy_user_id") != user_id:
    for key in ("mensagens", "documentos", "indexed_uploads", "failed_uploads"):
        st.session_state.pop(key, None)
    st.session_state.studybuddy_user_id = user_id
st.session_state.setdefault("mensagens", [])
st.session_state.setdefault("indexed_uploads", set())
st.session_state.setdefault("failed_uploads", set())


@st.cache_resource
def carregar_modelo():
    chave = os.getenv("GOOGLE_API_KEY")
    if not chave:
        raise RuntimeError("GOOGLE_API_KEY não encontrada. Cole sua chave do Google AI Studio no arquivo .env (veja a seção 2 do README).")
    return ChatGoogleGenerativeAI(model=chat_model_name(), google_api_key=chave, temperature=0.3)


@st.cache_resource
def carregar_memoria():
    return create_memory_graph()


@st.cache_resource
def carregar_store():
    return get_vector_store()


@st.cache_resource
def carregar_agente():
    return build_agent(carregar_modelo(), carregar_store())


def refresh_documents() -> None:
    st.session_state.pop("documentos", None)


try:
    memory_graph = carregar_memoria()
    st.session_state.mensagens = load_messages(memory_graph, user_id)
except Exception as exc:
    memory_graph = None
    st.error(f"Não foi possível abrir o histórico da conversa: {exc}")

store = None
try:
    store = carregar_store()
    if "documentos" not in st.session_state:
        # Consulta o Chroma só quando a lista muda, não a cada interação.
        st.session_state.documentos = list_user_documents(user_id, store)
except Exception as exc:
    st.error(f"Não foi possível abrir os materiais salvos: {exc}")
saved_documents: dict[str, str] = st.session_state.get("documentos", {})

with st.sidebar:
    st.header("📁 Materiais de estudo")
    uploads = st.file_uploader("Adicione seus materiais", type=["pdf", "txt"], accept_multiple_files=True)
    if uploads and store is not None:
        for upload in uploads:
            content = upload.getvalue()
            key = getattr(upload, "file_id", None) or (upload.name, hashlib.sha256(content).hexdigest())
            document_id = stable_document_id(user_id, upload.name, content)
            if key in st.session_state.indexed_uploads or key in st.session_state.failed_uploads:
                continue
            if document_id in saved_documents:
                st.session_state.indexed_uploads.add(key)
                continue
            # Cada arquivo é tratado separadamente: uma falha não impede os outros.
            try:
                with st.spinner(f"Indexando {upload.name}..."):
                    docs = load_uploaded_file(upload.name, content)
                    chunks = split_documents(docs)
                    count = index_file(user_id, upload.name, content, chunks, store)
                st.session_state.indexed_uploads.add(key)
                saved_documents[document_id] = upload.name
                st.success(f"{upload.name}: {count} trecho(s) indexado(s).")
            except Exception as exc:
                # Evita repetir a mesma falha (e gastar cota) a cada interação.
                st.session_state.failed_uploads.add(key)
                st.error(f"{upload.name}: não foi possível processar o arquivo. {friendly_error(exc)} "
                         "Para tentar de novo, remova o arquivo do campo acima e envie-o outra vez.")

    if saved_documents:
        st.markdown("**Materiais disponíveis**")
        for document_id, name in sorted(saved_documents.items(), key=lambda item: item[1].casefold()):
            col_name, col_delete = st.columns([5, 1])
            col_name.markdown(f"📄 {md_escape(name)}")
            if col_delete.button("🗑️", key=f"del-{document_id}", help=f"Remover {name}"):
                try:
                    delete_document(user_id, document_id, store)
                    refresh_documents()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Não foi possível remover {name}: {exc}")
    st.caption("Os materiais ficam vinculados ao endereço desta página. Guarde-o em privado para acessá-los após atualizar.")
    st.divider()
    with st.expander("Como o ClassMate responde"):
        st.write("Com materiais disponíveis, consulta trechos do documento. Pedidos como \"faça um quiz sobre X\" "
                 "usam a ferramenta de estudo quando há um único arquivo. Sem materiais, responde como tutor geral.")
    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        try:
            if memory_graph is not None:
                clear_messages(memory_graph, user_id)
            st.session_state.mensagens = []
            st.rerun()
        except Exception as exc:
            st.error(f"Não foi possível limpar a conversa: {exc}")

for message in st.session_state.mensagens:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.popover("＋  Ferramentas de estudo"):
    if saved_documents and store is not None:
        with st.form("study_tools"):
            task = st.selectbox("O que você quer fazer?", list(TASKS))
            chosen_id = st.selectbox("Material", list(saved_documents), format_func=lambda item: saved_documents[item])
            topic = st.text_input("Tema específico (opcional)")
            submitted = st.form_submit_button("Gerar", use_container_width=True)
        if submitted:
            try:
                with st.spinner("Preparando atividade..."):
                    result = generate_study_material(task, user_id, chosen_id, store, carregar_modelo(), topic=topic)
                title = f"{task} de {saved_documents[chosen_id]}" + (f" — {topic.strip()}" if topic.strip() else "")
                answer = f"**{md_escape(title)}**\n\n{result['content']}"
                if result["sources"]:
                    answer += "\n\n**Trechos consultados:**\n" + "\n".join(f"- {md_escape(source)}" for source in result["sources"])
                request = f"Gerar {title}"
                if memory_graph is not None:
                    save_turn(memory_graph, user_id, request, answer)
                else:
                    st.session_state.mensagens.extend([
                        {"role": "user", "content": request},
                        {"role": "assistant", "content": answer},
                    ])
                st.rerun()
            except Exception as exc:
                st.error(f"Não foi possível gerar a atividade: {friendly_error(exc)}")
    else:
        st.info("Envie um PDF ou TXT na barra lateral para usar estas ferramentas.")

pergunta = st.chat_input("Digite sua dúvida...")
if pergunta:
    previous_messages = list(st.session_state.mensagens)
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)
    with st.chat_message("assistant"):
        try:
            with st.spinner("ClassMate está pensando..."):
                agent = carregar_agente() if store is not None else None
                result = run_agent(pergunta, user_id, saved_documents, previous_messages,
                                   carregar_modelo(), store, agent=agent)
            check = evaluate_result(result)
            texto = result["answer"]
            if result["sources"]:
                texto += "\n\n**Fontes consultadas:**\n" + "\n".join(f"- {md_escape(source)}" for source in result["sources"])
            st.markdown(texto)
            with st.expander("Ver verificação da resposta"):
                route_label = ROUTE_LABELS.get(result.get("route", ""), "desconhecido")
                st.write(f"Caminho usado: {route_label}. Fontes consultadas: {check['source_count']}.")
                if check["issues"]:
                    for issue in check["issues"]:
                        st.warning(issue)
                else:
                    st.caption("Verificações de formato aprovadas. O conteúdo factual deve ser conferido no material.")
            st.session_state.mensagens.append({"role": "assistant", "content": texto})
            if memory_graph is not None:
                try:
                    save_turn(memory_graph, user_id, pergunta, texto)
                except Exception as exc:
                    st.warning(f"A resposta foi exibida, mas não pôde ser salva no histórico: {exc}")
        except Exception as exc:
            # Remove a pergunta sem resposta para não deixar o histórico inconsistente.
            st.session_state.mensagens.pop()
            st.error(f"Erro ao consultar o ClassMate: {friendly_error(exc)}")
