from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and return the result as a string."""
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"
    return str(result)

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""
    data = {
        "Brazil": "Brasília",
        "Paraguai": "Assunção",
        "Venezuela": "Caracas",
        "Colômbia": "Bogotá",
        "Equador": "Quito",
        "Canadá": "Ottawa",
        "Argentina": "Buenos Aires",
        "Bolívia": "La Paz",
        "Chile": "Santiago",
        "United States": "Washington, D.C."
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "Não tenho informações necessárias para responder sua pergunta."


llm = ChatOpenAI(model="gpt-5-mini", disable_streaming=True)
tools = [calculator, web_search_mock]

prompt = PromptTemplate.from_template(
"""
Answer the following questions as best you can. You have access to the following tools.
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for Observation.
- Never search the internet. Only use the tools provided.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)
agent_chain = create_react_agent(llm, tools, prompt, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors="Invalid format. Either provide an Action with Action Input, or a Final Answer only.",
    max_iterations=3)

print(agent_executor.invoke({"input": "Qual é a capital da França?"}))
print(agent_executor.invoke({"input": "Quantos clientes temos em 2024?"}))
print(agent_executor.invoke({"input": "Você acha isso bom ou ruim?"}))
print(agent_executor.invoke({"input": "Quantos clientes temos em 2024?"}))