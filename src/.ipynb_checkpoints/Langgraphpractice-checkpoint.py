#Another Langgraph example of a LinkedIN post generator and reviewer that makes use of the LLM

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

#Initialize the local LLM via chatollama

llm = ChatOllama(model="llama3.2", temperature=0.3)

#Define the Application state

class AgentState(TypedDict):

    topic: str
    draft:str
    critique: str
    revision_count:int

#define the Nodes

def writer_node(state:AgentState):
    print(f"-------The LLM i.e the writer is about to generate the version : {state['revision_count']+1} initial post-----------")
    system_msg = SystemMessage(content="You are an expert LinkedIn Content Creator known for high engagement and professional storytelling.")
    if state.get("critique"):
        user_content = f"""
        Topic: {state["topic"]}
        Previous Draft: {state['draft']}
        Feedback: {state['critique']}
        Please rewrite the post to fully incorporate the feedback that was provided
        """
    else:
        user_content = f"Write a compelling LinkedIn post about {state['topic']}. Include a strong hook and 3 hashtags."
    response = llm.invoke([system_msg, HumanMessage(content=user_content)])

    return {
        "draft":response.content,
        "revision_count": state["revision_count"]+1        
    }

def reflector_node(state:AgentState):
    print(f"The Agent or LLM is about to reflect on the quality")
    system_msg = SystemMessage(content="You are a critical Content Strategist. Your goal is to find flaws in LinkedIn posts to make them more viral.")
    user_content = f"""
    Critique this draft linkedin post and provide 2-3 actionable improvements:
    {state['draft']}
    """
    response = llm.invoke([system_msg, HumanMessage(content=user_content)])
    return {"critique": response.content}

# 4. Define the Logic (The Router)
def should_continue(state: AgentState):
    # We will stop after 2 revisions to balance quality and local compute time
    if state["revision_count"] >= 2:
        return "end"
    return "continue"

# 5. Build the Graph

workflow = StateGraph(AgentState)

workflow.add_node("writer", writer_node)
workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("writer")
workflow.add_edge("writer", "reflector")

workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "continue": "writer",
        "end": END
    }
)

# 6. Compile and Execute
app = workflow.compile()

# Start the process with your topic
initial_input = {
    "topic": "The shift from SaaS to 'Agentic AI' platforms in 2026", 
    "revision_count": 0
}

print("Starting Agentic Workflow...")

for event in app.stream(initial_input):
    for node, value in event.items():
        if "draft" in value:
            print(f"\n--- FINAL DRAFT (Version {value['revision_count']}) ---\n")
            print(value['draft'])
        if "critique" in value:
            print(f"\n--- REFLECTOR FEEDBACK ---\n")
            print(value['critique'])



    
#----------------------------------------------------------------------------------------------------------------
#Langgraph example that uses the LLM to act as the brain

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM

#1. Initialize the LLM
llm =OllamaLLM(model="llama3.2")

#2. Define the state
class AgentState(TypedDict):
    input_txt:str
    translated_txt:str
    feedback:str # this is the text from the LLM

#3. The LLM node(Brain)
def translator_node(state:AgentState):
    #this is the node or function that translates the text using the LLM
    print("-----The LLM is working------")

    # We combine the original text from the user with feedback from the critic
    prompt = f"""
    Translate the following to French: {state['input_txt']}
    Previous feedback: {state['feedback']}
    Please only return the translation
    """

    response=llm.invoke(prompt)
    return {"translated_txt":response.strip()}

#4. The critic node (The Judge)

def critic_node(state:AgentState):
    print("---------The Critic is grading-------")
    #if the feedback from the LLM was less than 3 words, it needs to do more work :-)
    if len(state['translated_txt'].split()) <1:
        print(f"The count of the split is {len(state['translated_txt'].split())}")
        return {"feedback": "Too short. Provide a full sentence"}
        
    return {"feedback": "good"}

#5. The Router

def should_continue(state:AgentState):
    if state['feedback'] == "good":
        return "end"
    return "retry"

#6 Build the graph

workflow = StateGraph(AgentState)

#7 Add the nodes

workflow.add_node("translator", translator_node)
workflow.add_node("critic",critic_node)

#8 Add the edges and conditional edges

workflow.add_edge(START, "translator")
workflow.add_edge("translator", "critic") #Go to critic after translating

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "retry": "translator",
        "end": END
    }
)

# Now compile the app
app=workflow.compile()

# Final step is to run it.

final_output = app.invoke({
    "input_txt": "Hello Friend",
    "translated_txt": "",
    "feedback": ""
})

prnt(f"\n Final AI translation:{final_output['translated_txt']}")


#-----------------------------------------------------------------------------------------------

#Simple example to highlight LangGraph's architecture.

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#1. Our Shared Notebook
class AgentState(TypedDict):
    draft: str
    iterations: int

#2. The writer node i.e adds a word to the draft
def writer_node(state:AgentState):
    print(f"Writer iterations: {state['iterations']}")
    new_draft = state['draft'] + 'word'
    return {"draft":new_draft, "iterations": state['iterations']+1}

#3. The critic logic i.e the Router
def critic_logic(state:AgentState):
    if len(state['draft']) <15:
        print("The draft is less than the required 15 words, so it needs to loop again")
        return "continue"
    
    print("The critic says, the draft looks good")
    return "end"

#4. Build the graph
builder = StateGraph(AgentState)

#5 Add the node(s)
builder.add_node("writer",writer_node)

#6. Start at the Writer
builder.add_edge(START, "writer")

#7. After the writer finishes the critic_logic decides next moves
builder.add_conditional_edges(
    "writer",
    critic_logic,
    {
        "continue": "writer",
        "end": END
    }
)

#8. Run the program

graph = builder.compile()

final_state = graph.invoke({"draft":"START", "iterations":0})

print(f"The final state result is {final_state['draft']}")
