import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from search import search

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")

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
    results = search(question, k=10)
    context = "\n\n".join(doc.page_content for doc, _score in results)
    prompt = prompt_template.format(context=context, question=question)
    return llm.invoke(prompt).content


if __name__ == "__main__":
    print("Chat iniciado. Digite 'sair' para encerrar.\n")
    while True:
        question = input("PERGUNTA: ").strip()
        if not question:
            continue
        if question.lower() == "sair":
            break
        print(f"RESPOSTA: {ask(question)}\n")
