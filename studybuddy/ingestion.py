"""Leitura e segmentação de arquivos PDF e TXT para o RAG."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_uploaded_file(filename: str, content: bytes) -> list[Document]:
    """Extrai páginas de PDF ou texto de TXT mantendo metadados de origem."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))
            docs = [
                Document(
                    page_content=(page.extract_text() or "").strip(),
                    metadata={"source": Path(filename).name, "page": index},
                )
                for index, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise ValueError(f"Não foi possível ler o PDF {filename}: {exc}") from exc
        docs = [doc for doc in docs if doc.page_content]
    elif suffix == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"O TXT {filename} precisa estar codificado em UTF-8.") from exc
        docs = [Document(page_content=text.strip(), metadata={"source": Path(filename).name})]
        docs = [doc for doc in docs if doc.page_content]
    else:
        raise ValueError("Formato não suportado. Envie um arquivo PDF ou TXT.")

    if not docs:
        raise ValueError(f"Nenhum texto extraível foi encontrado em {filename}.")
    return docs


def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """Divide documentos preservando source/page e adiciona índice do trecho."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Use chunk_size > chunk_overlap >= 0.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
    return chunks


def stable_document_id(user_id: str, filename: str, content: bytes) -> str:
    """ID determinístico para permitir reenvio sem duplicar o arquivo."""
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(filename).name)
    digest = sha256(content).hexdigest()
    return f"{user_id}:{safe_name}:{digest}"
