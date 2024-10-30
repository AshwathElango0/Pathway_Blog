# test_litellm.py

import os
from litellm import LLM

LLM_API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"

if not LLM_API_KEY:
    raise ValueError("Set the LLM_API_KEY environment variable.")

llm = LLM(model_name="gpt-3.5-turbo", api_key=LLM_API_KEY)

prompt = "What is the capital of France?"
response = llm(prompt)
print(response)
