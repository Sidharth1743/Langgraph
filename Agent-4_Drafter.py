from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,SystemMessage,BaseMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

document_content = ""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_tool(content:str)-> str:
    """Updates the document with the provided content """
    global document_content
    document_content = content
    return f"Document has been updated successfully: Current content : \n{document_content}"

@tool
def save(filename:str) -> str:
    """Save the current document to a text file with suitable name and terminate the process
    Args:
        filename : Name for the text file
    """
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n Document has been saved succcessfully {filename}")
        return f"Document has been successfully to {filename}"
    except Exception as e:
        return f"Error saving the file :{str(e)}"

tools = [update_tool , save]

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash").bind_tools(tools)

def agent(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
    f"""You are a helpful Drafter assistant. Your role is to help the user update and save the file. 
        - If the user wants to update or modify the content then use the 'update' tool with the complete update content.
        - If the user wants to save and finsih,you need to use the 'save' tool with the full content.
        - Make sure to show the current document state after modifications.
        The current document content is : {document_content}
        - If the document is empty then create a new one
         """
        )
    if not state['messages']:
        user_input = "I'm ready to help you update a document.What would you are going to create?"
        user_message = HumanMessage(content = user_input)

    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\n USER: {user_input}")
        user_message= HumanMessage(content = user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response,"tool_calls") and response.tool_calls:
        print(f"USING THE TOOL: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": list(state["messages"]) + [user_message , response]}

def should_continue(state:AgentState)-> str:
    """Determine if we should continue or end the conversation"""
    messages = state["messages"]
    if not messages :
        return "continue"

    for message in reversed(messages):
        if (isinstance(message,ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower()):
            return "end"
    return "continue"

def print_message(messages):
    """Function to make the content more readable format"""
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message,ToolMessage):
            print(f"\nTOOL RESULT : {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent",agent)
graph.add_node("tools",ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"agent",
        "end":END,
    }
)

app = graph.compile()

def run_doc_agent():
    print("\n ===== DRAFTER =====")
    state = {"messages":[]}
    for step in app.stream(state,stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])
    print("\n ====== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_doc_agent()


