# Roteiro de apresentação do ClassMate AI

## Problema e proposta

Estudantes possuem materiais extensos e precisam localizar conceitos, revisar conteúdo e praticar. O ClassMate transforma PDFs e TXT em uma base consultável e oferece chat e atividades de estudo.

## Demonstração sugerida

1. Mostrar a interface e enviar um PDF com texto selecionável.
2. Perguntar um fato presente no arquivo; apontar a fonte e a página.
3. Perguntar algo ausente; mostrar a resposta de abstenção.
4. Pedir um quiz no chat e gerar flashcards na barra lateral.
5. Atualizar a página; mostrar que o arquivo e o histórico continuam disponíveis.
6. Abrir uma URL nova; mostrar a separação das conversas locais.

## Explicação técnica curta

O `pypdf` extrai texto, o LangChain divide em trechos, embeddings Gemini transformam texto em vetores e o Chroma recupera trechos semelhantes. O LangGraph direciona a pergunta à ferramenta adequada e registra a conversa no SQLite. O Streamlit apresenta tudo ao usuário.

## Limites a declarar

O sistema depende da API Gemini e de suas cotas. PDFs escaneados exigem OCR. As fontes mostram trechos consultados, e respostas de IA exigem conferência humana. O acesso por URL é adequado à demonstração local; uma versão pública exigiria autenticação.
