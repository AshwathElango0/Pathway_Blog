from typing import Annotated, Literal
from agent import Agent, ChatResult
import logging

# Configure root logger (optional)
logging.disable(logging.CRITICAL)
Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    """Simple calculator function that takes two numbers and an operator and returns the result."""
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")


instructions = """You are a helpful AI assistant. 
    You can help with simple calculations.
    Return 'exit' when the task is done."""

LLM_API_KEY = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'

calculator_agent = Agent.from_model("gemini/gemini-1.5-flash", name="Assistant", instructions=instructions, api_key=LLM_API_KEY, temperature=0.0, max_tokens=10000)

user_proxy = Agent.from_model("gemini/gemini-1.5-flash",  name="UserProxy", api_key=LLM_API_KEY, temperature=0.0, max_tokens=10000)

calculator_agent.add_tools([calculator])

user_proxy.register_tool_for_execution(calculator)
import json

chat_result = user_proxy.initiate_chat(calculator_agent, max_turns=5, user_input="What is 23*78 - 53?", termination_cond=lambda x: isinstance(x, str) and "exit" in x.lower())

# print(chat_result)