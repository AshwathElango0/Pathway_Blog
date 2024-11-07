from typing import Annotated, Literal
from agent import Agent, ChatResult
import logging

# Configure root logger (optional)
logging.disable(logging.CRITICAL)

LLM_API_KEY = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'

student_agent = Agent.from_model(
    model="gemini/gemini-1.5-flash",
    name="Student_Agent",
    instructions="You are a student willing to learn.",
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=10000,
)
teacher_agent = Agent.from_model(
    model="gemini/gemini-1.5-flash",
    name="Teacher_Agent",
    instructions="You are a math teacher.",
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=10000,
)


chat_result = student_agent.initiate_chat(teacher_agent, max_turns=2, user_input="What is Triangle Inequality?")

# print(chat_result)