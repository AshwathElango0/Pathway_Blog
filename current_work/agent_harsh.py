from agent import Agent
from typing import List

api_key = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'

instructions = """You are a helpful assistant.
"""


agent = Agent.from_model("gemini/gemini-1.5-flash",
        name="Assitant",
        instructions=instructions, 
        api_key=api_key, 
        temperature=0.0, 
        max_tokens=10000
    )


import json
while True:
    user_query = input("Enter your query: ")
    res = agent.send_request(user_query)
    print(res)