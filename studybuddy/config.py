"""Configuração da aplicação e dos modelos.

Os caminhos são resolvidos a partir da raiz do projeto, e não da pasta de onde
o Streamlit foi iniciado, para que os dados fiquem sempre em ``data/``.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
MEMORY_DB = DATA_DIR / "conversations.sqlite"

COLLECTION_NAME = "studybuddy_documents"
DEFAULT_CHAT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def chat_model_name() -> str:
    return os.getenv("GEMINI_MODEL") or DEFAULT_CHAT_MODEL


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
