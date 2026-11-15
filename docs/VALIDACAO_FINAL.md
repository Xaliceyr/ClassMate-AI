# Validação completa — executar após as Fases 5 e 6

## Preparação

1. Na raiz do projeto, ative `.venv` e execute `python -m pip install -r requirements.txt`.
2. Confirme `GOOGLE_API_KEY`, `GEMINI_MODEL` e `EMBEDDING_MODEL` no `.env` sem divulgar a chave.
3. Execute `python -m pytest -q` (o teste real do Gemini só roda com `RUN_GEMINI_TESTS=1`).
4. Inicie `streamlit run app.py`.

## Roteiro manual

| Caso | Ação | Resultado esperado |
|---|---|---|
| Chat geral | Pergunte algo sem enviar arquivo | Resposta didática, sem alegar consulta a PDF |
| Upload | Envie PDF com texto e TXT UTF-8 | Arquivos aparecem em Materiais disponíveis |
| RAG | Pergunte fato específico do PDF | Resposta e nome/página do arquivo |
| Ausência | Pergunte algo fora do material | Declaração de ausência, sem fonte inventada |
| Continuação | Faça uma pergunta de seguimento | Conversa recente ajuda a interpretar a pergunta |
| Ferramentas | Gere resumo, quiz, flashcards e explicação simples | Atividades aparecem no histórico com trechos consultados |
| Agente | Com um arquivo, peça “faça um quiz” no chat | Ferramenta de estudo é selecionada |
| Vários arquivos | Envie dois arquivos e use a barra lateral | Atividade usa somente o arquivo escolhido |
| Persistência | Atualize a página e reinicie o Streamlit | Materiais e mensagens reaparecem na mesma URL |
| Limpeza | Clique Limpar conversa | Mensagens somem; materiais continuam disponíveis |
| Remoção | Clique 🗑️ ao lado de um material | O arquivo some da lista e das respostas |
| Falso comando | Com um arquivo, pergunte "o que diz o resumo do artigo?" | Rota de consulta ao material, não atividade |
| Isolamento | Abra URL nova sem `studybuddy_id` | Histórico e materiais da URL anterior não aparecem |
| Quota | Se ocorrer 429 de embeddings | Interface mostra erro legível ou espera a nova tentativa |

Registre capturas da pergunta, resposta, página citada, atividades e persistência. Não publique a URL completa nem o `.env` nas capturas: a URL funciona como acesso aos dados locais.
