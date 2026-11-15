"""Teste de integração com a API Gemini real.

É ignorado por padrão para que `pytest` funcione sem chave e sem gastar cota.
Para executá-lo: defina RUN_GEMINI_TESTS=1 e configure GOOGLE_API_KEY no .env.
"""

import os

import pytest
from dotenv import load_dotenv

from studybuddy.config import BASE_DIR, chat_model_name

load_dotenv(BASE_DIR / ".env")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_GEMINI_TESTS") != "1", reason="Defina RUN_GEMINI_TESTS=1 para chamar a API Gemini.")
def test_gemini_connection():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "chave_ficticia_exemplo":
        pytest.skip("GOOGLE_API_KEY não configurada no .env")
    modelo = ChatGoogleGenerativeAI(model=chat_model_name(), google_api_key=api_key, temperature=0)
    resposta = modelo.invoke("Responda apenas: ClassMate AI conectado com sucesso!")
    assert resposta.content
