import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate

load_dotenv()
for key in ("OPENAI_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(key):
        raise RuntimeError(f"Environment variable {key} is not set")

embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL", "text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

llm = ChatOpenAI(model="gpt-4o-mini")

prompt_template = PromptTemplate.from_template("""CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO\"""")


def ask(question: str) -> str:
    docs = store.similarity_search(question, k=10)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    print("Chat iniciado. Digite 'sair' para encerrar.\n")
    while True:
        question = input("Você: ").strip()
        if not question:
            continue
        if question.lower() == "sair":
            break
        print(f"Assistente: {ask(question)}\n")
