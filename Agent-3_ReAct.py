from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
from langchain_core import agents
from langchain_core.messages import BaseMessage , ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph import graph
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int , b:int):
    """This is an addition function that adds 2 numbers """
    return a + b

@tool
def multiply(a:int , b:int):
    """This is an multiplication function that multiplies the numbers"""
    return a * b

@tool
def division(a:int , b:int):
    """This is a division function that divides two numbers"""
    return a / b

tools =[add , multiply , division]

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash").bind_tools(tools)


def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent",model_call)
 
tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools","agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 6 + 9 and multiply the result by 100.")]}
print_stream(app.stream(inputs , stream_mode="values"))