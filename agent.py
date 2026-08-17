from dotenv import load_dotenv
load_dotenv()
import asyncio
from pathlib import Path
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


# Carga de archivos de memoria
memory = Path("memory/MEMORY.md").read_text()
context = Path("memory/contexto_equipo.md").read_text()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# Nodo principal asíncrono (requerido por LangGraph Studio)
async def agent(state: State):
    system_prompt = SystemMessage(
        content=f"""You are a test agent migrated from Letta.

Here is the imported Letta memory:

=== MEMORY ===
{memory}

=== CONTEXT ===
{context}

Only use information contained in this imported data.
"""
    )

    messages = [system_prompt] + list(state["messages"])
    response = await model.ainvoke(messages)

    return {"messages": [response]}


# Construcción del grafo
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile()


# Función principal de prueba para la terminal
async def main():
    result = await app.ainvoke({
        "messages": [
            (
                "user",
                """From the imported Letta memory:

1. Who are Carlos, Iván, Félix and Yomar?
2. What information is stored about them?
3. Are they actually users represented in the imported memory?
4. What was your last conversation about with Yomar and Carlos?
5. Are there any conversations or recorded interactions with other team members like Iván or Félix, and can you tell me excatly one conversation?
6. Is it okay for you to tell me about others conversations?
7. Was what you shared with me before okay?

Do not guess. Only use the imported files."""
            )
        ]
    })
    print("\n=== RESPUESTA DE LA TERMINAL ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())