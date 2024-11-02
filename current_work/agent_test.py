from agent import Agent
from typing import List
api_key = 'AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc'


instructions = """You are a helpful assistant, capable of retrieving and summarizing information by selectively using available functions, each designed to respond differently based on the nature of the user's question. Remember, you possess no independent knowledge, and any response should either come from the user’s provided information or the functions you call. Here is how to decide which function to use:

    Use function1: If the user's query implies or directly states that the answer can be found solely within their uploaded documents, choose function1. This is true even if there’s only a slight indication that the user’s uploaded information may answer their question. Do not use external sources in this case.

    Use function2: If the query suggests that an answer might require both the user’s documents and additional information inferred from other sources, or if it lacks sufficient detail to answer entirely on its own, choose function2. This function combines insights from the provided documents and external data that you can infer or deduce.

    Use function3: Use it for questions whose answer lie explicitly in the query itself. So, factual questions, definitions will never use function3.

Always select the function that fits best according to these guidelines, and rely on your own response only if calling function3.
"""


agent = Agent.from_model("gemini/gemini-1.5-flash", instructions=instructions, api_key=api_key, temperature=0.0, max_tokens=50)

def function1(uploaded_doc_names: List[str]):
    """Function to perform if the given user query can be answered only by the documents uploaded by him"""
    return "This is the response from function1"

def function2(uploaded_doc_names: List[str]):
    """Function to perform if the given user query can be answered by using user uploaded documents and external sources, external sources might not be mentioned but must be inferred."""
    return "This is the response from function2"

def function3(string: str):
    """Function to perform if the given user query doesn't need any sources, and can be answered by the agent itself"""
    return "This is the response from function3"

agent.add_tools([function1, function2, function3])
# set env

LLM_API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"
from custom_llm_chat import CustomLiteLLMChat

model = CustomLiteLLMChat(
    model="gemini/gemini-1.5-flash",
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=100,
)

while True:
    user_query = input("Enter your query: ")
    response = agent.send_request(user_query)
    print(response)
    if response.function.name == 'function3':
        res = model.get_response([{"role": "user", "content": user_query}])
        print(f'\n\nResponse: {res}')

    
