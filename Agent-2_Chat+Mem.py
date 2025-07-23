
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langgraph import graph
from langgraph.graph import START,END,StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def process(state:AgentState) -> AgentState:
    response=llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI:{response.content}")
    return state

graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()

conversation_history = []
user_input = input("Enter: ")
while user_input!="exit":
    conversation_history.append(HumanMessage(content=user_input))
    result=agent.invoke({"messages":conversation_history})
    conversation_history=result["messages"]
    user_input=input("Enter: ")
with open("logging.txt" ,"w") as file:
    file.write("Your conversation Log:\n")
    for message in conversation_history:
        if isinstance(message,HumanMessage): 
            file.write(f"\n You:{message.content}")
            print("\n")
        elif isinstance(message,AIMessage):
            file.write(f"\nAI:{message.content}")
    file.write("End of conversation")

print("COnversation History saved to logging.txt")