from agent import Agent
from typing import List

api_key = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'

instructions = """You are a helpful assistant. Return 1, 2, or 3 based on the following:
1 - Use it when answer can be provided solely based on uploaded documents. It must be implied that query requires answer from uploaded documents.
2 - Use it when question is too generic, and you need to use external sources
3 - Answer can be provided directly
"""


agent = Agent.from_model("gemini/gemini-1.5-flash", instructions=instructions, api_key=api_key, temperature=0.0, max_tokens=10000, name="Assistant")


from typing import Literal
def choose_path(path: Literal[1,2, 3]) -> str:
    """Called based on what path is decided.
        1 - Use it when answer can be provided solely based on uploaded documents. It must be implied that query requires answer from uploaded documents.
        2 - Use it when question is too generic, and you need to use external sources
        3 - Answer can be provided directly
    """
    if path == 1:
        # Only use uploaded documents
        return "This is the response based solely on uploaded documents."
    elif path == 2:
        # Use both uploaded documents and external sources
        return "This is the response based on uploaded documents and inferred external sources."
    elif path == 3:
        # Answer can be provided directly
        return "This is the response that can be answered directly without retrieval."

LLM_API_KEY = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'
from custom_llm_chat import CustomLiteLLMChat

model = CustomLiteLLMChat(
    model="gemini/gemini-1.5-flash",
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=1000,
)

agent.add_tools([choose_path])
import json
while True:
    user_query = input("Enter your query: ")
    res = agent.send_request(user_query)
    choices = res.choices[0]["message"]["tool_calls"]
    # print(choices[0].function.arguments)
    if len(choices) ==0:
        path = 2
    else:
        args = choices[0].function.arguments
        args = json.loads(args)
        path = int(args["path"])
    response = choose_path(path)
    print(response)
    
    if path == 3:
        res = model.get_response([{"role": "user", "content": user_query}])
        print(f'\n\nResponse: {res}')
