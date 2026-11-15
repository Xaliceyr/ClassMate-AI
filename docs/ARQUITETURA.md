# Arquitetura do ClassMate AI

## Fluxo principal

1. `app.py` recebe PDF/TXT no Streamlit.
2. `studybuddy/ingestion.py` extrai texto, preserva arquivo e página e divide em trechos.
3. `studybuddy/rag.py` gera embeddings Gemini, grava no Chroma persistente em `data/chroma` e busca trechos com filtro `user_id`.
4. `studybuddy/agent.py` usa LangGraph para escolher entre consulta aos documentos, ferramenta de estudo e chat geral. Pedidos explícitos de resumo, quiz, flashcards e explicação simples usam o arquivo quando só há um material; com vários materiais, o usuário escolhe o arquivo na barra lateral.
5. `studybuddy/extras.py` gera atividades a partir de até 12 trechos do arquivo selecionado: os mais relacionados ao tema, quando há tema, ou uma amostra distribuída pelo arquivo.
6. `studybuddy/memory.py` grava cada pergunta e resposta em `data/conversations.sqlite` usando checkpoint SQLite do LangGraph.
7. `studybuddy/evaluation.py` verifica formato da resposta, presença de fontes e abstenção. Essas verificações não medem correção factual.

## Identidade e armazenamento

O identificador UUID na URL permite reencontrar documentos e histórico após atualizar a página. O Chroma aplica filtro por `user_id` em cada consulta. Isso é isolamento funcional para a demonstração local, não autenticação. Quem tiver a URL completa pode acessar os dados associados. Para produção, substitua o UUID da URL por um identificador obtido de login e aplique autorização no servidor.

`data/` e `.env` são locais e ignorados pelo Git. Os caminhos de `data/` são resolvidos a partir da raiz do projeto (`studybuddy/config.py`), independentemente da pasta de onde o Streamlit é iniciado. Cada conversa mantém apenas o checkpoint mais recente no SQLite, e o botão **Limpar conversa** apaga os checkpoints da conversa com `PRAGMA secure_delete` ativo. A indexação de um arquivo é tudo-ou-nada: se um lote de embeddings falhar, os trechos já gravados são removidos. Os PDFs originais não são armazenados: o Chroma conserva os trechos extraídos.

## Limitações conhecidas

- PDFs escaneados como imagem precisam de OCR.
- O modelo Gemini pode errar; fontes exibidas são trechos consultados, não uma prova automática de cada frase.
- O RAG busca por similaridade sem limite mínimo de relevância; a instrução ao modelo pede abstenção quando o conteúdo não sustenta resposta.
- Resumos e atividades sem tema em documentos extensos usam uma amostra; não cobrem necessariamente todas as páginas.
- A busca usa a pergunta anterior apenas quando a atual parece uma continuação (curta ou iniciada por "e", "isso", "ele" etc.).
- Embeddings estão sujeitos aos limites da API Google, incluindo respostas 429.
