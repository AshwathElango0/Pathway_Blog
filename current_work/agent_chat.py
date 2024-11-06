from typing import Annotated, Literal
from agent import Agent, ChatResult
import logging

# Configure root logger (optional)
logging.basicConfig(
    level=logging.INFO,  # Default level for root logger
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure 'requests' logger to DEBUG
requests_logger = logging.getLogger('requests')
requests_logger.setLevel(logging.DEBUG)

agent_logger = logging.getLogger('agent')
agent_logger.setLevel(logging.WARNING)

# Configure 'litellm' logger to WARNING (suppress DEBUG and INFO)
litellm_logger = logging.getLogger('litellm')
litellm_logger.setLevel(logging.WARNING)
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
    Return 'exit' when the task is done.
    Follow BODMAS rules for calculations.
    """

LLM_API_KEY = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'

calculator_agent = Agent.from_model("gemini/gemini-1.5-flash", name="A Simple Calculator", instructions=instructions, api_key=LLM_API_KEY, temperature=0.0, max_tokens=10000)

user_proxy = Agent.from_model("gemini/gemini-1.5-flash",  name="UserProxy", api_key=LLM_API_KEY, temperature=0.0, max_tokens=10000)

calculator_agent.add_tools([calculator])

user_proxy.register_tool_for_execution(calculator)
import json

chat_result = user_proxy.initiate_chat(calculator_agent, user_input="What is 44232 + 13312 / (232 - 32)?", termination_cond=lambda x: "exit" in x.lower())

print(chat_result)