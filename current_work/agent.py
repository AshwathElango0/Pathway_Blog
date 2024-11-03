import pydantic
from typing import List, Union, Dict, Any, Optional, Callable

import pathway
import pydantic


import copy
import json
import logging
from typing import List, Dict, Optional

import litellm  # Ensure litellm is installed: pip install litellm
import pathway as pw
from pathway.udfs import AsyncRetryStrategy, CacheStrategy
from pathway.xpacks.llm.llms import BaseChat
from pathway.xpacks.llm._utils import _check_model_accepts_arg  # Ensure this utility is correctly imported
from litellm.utils import function_to_dict
# Configure logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class LLMAgent(BaseChat):
    """
    Custom Pathway wrapper for LiteLLM Chat services using litellm.

    Inherits from BaseChat and implements the __wrapped__ method for integration with Pathway.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        retry_strategy: Optional[AsyncRetryStrategy] = None,
        cache_strategy: Optional[CacheStrategy] = None,
        model: Optional[str] = None,
        **litellm_kwargs,
    ):
        super().__init__(
            executor=pw.udfs.async_executor(capacity=capacity, retry_strategy=retry_strategy),
            cache_strategy=cache_strategy,
        )
        self.litellm_kwargs = litellm_kwargs.copy()
        if model is not None:
            print(f"Setting model to {model}")
            self.litellm_kwargs["model"] = model

    def _check_model_accepts_arg(self, model: str, provider: str, arg_name: str) -> bool:
        """
        Wrapper to the standalone _check_model_accepts_arg function.
        """
        return _check_model_accepts_arg(model, provider, arg_name)

    def _accepts_call_arg(self, arg_name: str) -> bool:
        """
        Check whether the LLM accepts the argument during the inference.
        If `model` is not set, return `False`.
        """
        model = self.kwargs.get("model")
        if model is None:
            return False

        # Assuming the provider is 'litellm' for LiteLLM models
        provider = "litellm"  # Adjust if using different providers
        return self._check_model_accepts_arg(model, provider, arg_name)
    def __wrapped__(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Synchronously sends messages to LiteLLM and returns the response.

        Args:
            messages (List[Dict[str, str]]): List of messages with 'role' and 'content'.
            **kwargs: Override for litellm_kwargs.

        Returns:
            Optional[str]: LLM response or None.
        """
        # logger.info("Entering __wrapped__ method.")

        # Merge default and overridden kwargs
        request_kwargs = {**self.litellm_kwargs, **kwargs}
        # logger.debug(f"Request kwargs: {request_kwargs}")

        # Log the request event
        event = {
            "_type": "lite_llm_chat_request",
            "kwargs": copy.deepcopy(request_kwargs),
            "messages": messages,
        }
        # logger.info(json.dumps(event))

        try:
            # Call litellm completion synchronously
            response = litellm.completion(messages=messages, **request_kwargs)
            # logger.info("Received response from litellm.completion.")

            # Log the response event
            event = {
                "_type": "lite_llm_chat_response",
                "response": "hello",
            }
            # logger.info(json.dumps(event))
            return response
        except Exception as e:
            # logger.error(f"Error generating response from LiteLLM: {e}")
            return None

    def __call__(self, messages: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """
        Integrates with Pathway's UDF system.

        Args:
            messages (pw.ColumnExpression): Column containing message lists.
            **kwargs: Override for litellm_kwargs.

        Returns:
            pw.ColumnExpression: Column with LLM responses.
        """
        print("Inside __call__")
        return super().__call__(messages, **kwargs)
    
    def get_response(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Helper method to directly call the __wrapped__ method.
        """
        return self.__wrapped__(messages, **kwargs)
    


class ChatResult:
    """Placeholder for ChatResult. Implement as needed."""
    def __init__(self, history: List[Dict] = None):
        self.history = history or []


class Agent:
    name: str = "assistant"
    model : str 
    instructions: str = "You are a helpful agent"
    llm: LLMAgent = pydantic.Field(default=None, exclude=True)
    tools: List[Dict] = []
    
    def __init__(self, model, llm, name="assistant",instructions=None):
        self.model = model
        self.llm = llm
        self.instructions = instructions
        
    @classmethod
    def from_model(cls, model: str, name: str, instructions:str, **kwargs):
        llm = LLMAgent(model=model, **kwargs)
        return cls(model=model, llm=llm,name = name, instructions=instructions)
    
    def add_tools(self, tools: List[Callable]):
        for tool in tools:
            self.tools.append(function_to_dict(tool))
            
    def send_request(self, query):
        try:
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": query}
            ]

            response = self.llm.get_response(messages, tools=self.tools)
            print(response)
            
            return response
            
        except Exception as e:
            event = {
                "_type": "lite_llm_agent_error",
                "error": str(e)
            }
            
            # logger.error(json.dumps(event))
            raise e
        
        
    def initiate_chat(
        self,
        partner_agent: "Agent",
        user_input: str,
        max_turns: int = 5,
    ) -> ChatResult:
        """
        Initiate an interactive chat session between two agents.

        Args:
            partner_agent (Agent): The second agent to interact with.
            max_turns (Optional[int]): Maximum number of conversation turns. Defaults to unlimited.
            initial_message (Optional[str]): Initial message to start the chat. If None, prompts the user.

        Returns:
            ChatResult: The result of the chat session, including history.
        """
        print("Chat session initiated between two agents. Type 'exit' or 'quit' to end the chat.\n")

        chat_history = [{self.name: user_input}]

        for i in range(max_turns):
            # Agent 1 (self) sends a message to Agent 2 (partner_agent)
            print("---------------------------------------------------")
            print(f"{self.name} -> {partner_agent.name}: {user_input}")
            
            # Send the message to the partner agent
            response = partner_agent.send_request(user_input)
            print(response)
            
            # Save the response in the chat history
            
            choices = response.choices[0]
            message = choices["message"]
            tool_calls = message["tool_calls"]
            
            if len(tool_calls) == 0:
                print("No tool calls")
            else:
                print(tool_calls[0].function.arguments)
                

            chat_history.append({partner_agent.name: message})
            
            # Agent 2 (partner_agent) sends a message to Agent 1 (self)
            print("---------------------------------------------------")
            print(f"{partner_agent.name} -> {self.name}: {message}")
            
            # Send the message to the self agent
            response = self.send_request(message)
            print(response)
            
            # Save the response in the chat history
            choices = response.choices[0]
            message = choices["message"]
            
            chat_history.append({self.name: message})
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                break
                
            
            
            
            
    

