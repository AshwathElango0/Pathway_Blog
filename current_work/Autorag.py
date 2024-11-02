from autogen import Agent
from autogen import Tool
from typing import List
from custom_llm_chat import CustomLiteLLMChat

# API key for the LLM model
api_key = 'AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc'

# Instructions to guide the agent's function selection behavior
instructions = """
You are a helpful assistant who retrieves and summarizes information by selectively using available functions, each responding differently based on the user's question. You have no independent knowledge, and any response should come either from the user’s provided information or the functions you call.

- Use `function1` for questions implying that the answer is found solely within uploaded documents.
- Use `function2` when questions require a combination of user documents and inferred external information.
- Use `function3` only if the question itself contains the explicit answer.
"""

# Define each function with a description and selection criteria
def function1(uploaded_doc_names: List[str]):
    """This function responds if the query can be answered only by the user's uploaded documents."""
    return "This is the response from function1"

def function2(uploaded_doc_names: List[str]):
    """This function responds if the query requires both user-uploaded documents and inferred external information."""
    return "This is the response from function2"

def function3(string: str):
    """This function responds if the query contains all necessary information for the answer."""
    return "This is the response from function3"

# Wrap functions as AutoGen tools with clear descriptions
tools = [
    Tool(name="function1", function=function1, description="Only use this function if the answer is found solely in uploaded documents."),
    Tool(name="function2", function=function2, description="Use this function when an answer requires user documents plus inferred external information."),
    Tool(name="function3", function=function3, description="Use this only when the answer is explicit within the query."),
]

# Initialize the agent with the model, API key, and instructions
agent = Agent(
    model="gemini/gemini-1.5-flash",
    api_key=api_key,
    instructions=instructions,
    tools=tools,
    temperature=0.0,
    max_tokens=50,
)

# Set up a loop for processing user queries
while True:
    user_query = input("Enter your query: ")
    
    # Process the query through the agent
    response = agent.send_request(user_query)
    print(f"\nResponse: {response['content']}")

    # If `function3` was chosen, generate a direct answer using the LLM
    if response['function_called'] == 'function3':
        
        # Create a direct response using the language model for questions containing the answer
        model = CustomLiteLLMChat(
            model="gemini/gemini-1.5-flash",
            api_key=api_key,
            temperature=0.0,
            max_tokens=100,
        )
        
        # Generate response for the query itself
        direct_response = model.get_response([{"role": "user", "content": user_query}])
        print(f"\nDirect Response: {direct_response}")
