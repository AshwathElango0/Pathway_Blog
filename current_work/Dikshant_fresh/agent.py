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
import os

os.environ['LITELLM_LOG'] = 'DEBUG'
# Configure logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
os.environ['LITELLM_LOG'] = 'DEBUG'
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
            raise e
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
        
        if kwargs.get('tools') is not None:
            if len(kwargs.get('tools')) == 0:
                kwargs.pop('tools')
        return self.__wrapped__(messages, **kwargs)
    


class ChatResult:
    """Placeholder for ChatResult. Implement as needed."""
    def __init__(self, history: List[Dict] = None):
        self.history = history or []
    
    def generate_context(self):
        context = ""
        for message in self.history:
            for agent, content in message.items():
                context += f"{agent}:\n{content}\n"
        
        return context
    
    def add_message(self, agent: str, message: str):
        self.history.append({agent: message})
        
    def __str__(self):
        return self.generate_context()
    
    def __repr__(self):
        return self.generate_context()


class Agent:
    name: str = "assistant"
    model : str 
    instructions: str = "You are a helpful agent"
    llm: LLMAgent = pydantic.Field(default=None, exclude=True)
    tools: List[Dict] = []
    functions: Dict[str, Callable] = {}
    
    def __init__(self, model, llm, name="assistant",instructions=None):
        self.model = model
        self.llm = llm
        self.instructions = instructions
        self.name  = name

        
    @classmethod
    def from_model(cls, model: str, name: str, instructions:str = "", **kwargs):
        llm = LLMAgent(model=model, **kwargs)
        return cls(model=model, llm=llm,name = name, instructions=instructions)
    
    def add_tools(self, tools: List[Callable]):
        for tool in tools:
            self.tools.append(function_to_dict(tool))
    def register_tool_for_execution(self, tool: Callable):
        self.functions[tool.__name__] = tool
        
    def send_request(self, query):
        try:
            messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": query}
            ]

            response = self.llm.get_response(messages, tools=self.tools)
            
            return response
            
        except Exception as e:
            event = {
                "_type": "lite_llm_agent_error",
                "error": str(e)
            }
            
            # logger.error(json.dumps(event))
            raise e
        
        
    import json
    from typing import Optional, Tuple

# Assuming ChatResult and Agent classes are defined elsewhere
# from your_module import ChatResult, Agent

    def initiate_chat(
            self,
            partner_agent: "Agent",
            user_input: str,
            max_turns: Optional[int] = None,
            termination_cond: Optional[Callable[[str], bool]] = None,
            silent: bool = False
        ) -> "ChatResult":
        """
        Initiate an interactive chat session between two agents.

        Args:
            partner_agent (Agent): The second agent to interact with.
            user_input (str): Initial message to start the chat.
            max_turns (Optional[int]): Maximum number of conversation turns. Defaults to unlimited.
            termination_cond (Optional[Callable[[str], bool]]): A callable that takes the latest message and returns True to terminate the chat.
            silent (bool): Whether to suppress print statements. Default is False.

        Returns:
            ChatResult: The result of the chat session, including history.
        """
        self._log("Chat session initiated between two agents. Type 'exit' or 'quit' to end the chat.\n", silent)

        chat_history = self._initialize_chat_history(partner_agent, user_input)
        context = self._build_context(self.name, partner_agent.name, user_input)
        turn = 0
        while max_turns is None or turn < max_turns:
            # Check for termination conditio
            if turn == 0:
                self._log_separator(silent)
                self._log_message(f"\n{self.name} -> {partner_agent.name}:\n{user_input}\n", silent)
                self._log_separator(silent) 
            # Partner Agent's Turn
            err = False
            try:
                partner_response = self._send_message(partner_agent, context)
                latest_message, context, is_tool_output = self._process_partner_response(partner_response, partner_agent, chat_history, context, silent)
            except Exception as e:
                err = True
                latest_message = self._handle_error(partner_agent.name, f"Error during partner agent request: {str(e)}", chat_history, silent)
                
                print(f"Error: {latest_message}")
                break
                # Context remains unchanged in case of an error
                
            # Check for exit commands from partner agent
            if self._check_exit_condition(latest_message):
                self._log("Exit command received. Ending chat session.", silent)
                break
            
            if is_tool_output:
                # Skip self agent's turn if tool output is received
                context_addition = f"Output from tool execution={latest_message}"
                context = self._update_context(context, self.name, context_addition)
                self._log_separator(silent)
                self._log_message(f"\n{self.name} -> {partner_agent.name}:\n{context_addition}\n", silent)
                self._log_separator(silent)
                self._add_to_history(chat_history, self.name, context_addition)
                if termination_cond and termination_cond(latest_message):
                    self._log("Termination condition met. Ending chat session.", silent)
                    break
                turn += 1
                
                continue
            # Self Agent's Turn
            try:
                
                if termination_cond and termination_cond(latest_message):
                    self._log("Termination condition met. Ending chat session.", silent)
                    break
                
                self_response = self._send_message(self, context)
                message = self._extract_message_content(self_response)
                self._add_to_history(chat_history, self.name, message)
                context = self._update_context(context, self.name, message)                

                self._log_separator(silent)
                self._log_message(f"\n{self.name} -> {partner_agent.name}:\n{message}\n", silent)
                self._log_separator(silent)
                turn += 1
                
                
                # Update user_input for the next turn
                user_input = message
                
            except Exception as e:
                self._handle_error(self.name, f"Error during self agent request: {str(e)}", chat_history, silent)
                break
        return ChatResult(history=chat_history)

    # -------------------- Helper Methods --------------------

    def _log(self, message: str, silent: bool):
        """
        Log a message to the console if not silent.

        Args:
            message (str): The message to log.
            silent (bool): Whether to suppress logging.
        """
        if not silent:
            print(message)

    def _log_separator(self, silent: bool):
        """
        Print a separator line if not silent.

        Args:
            silent (bool): Whether to suppress logging.
        """
        if not silent:
            print("---------------------------------------------------")

    def _log_message(self, message: str, silent: bool):
        """
        Log a chat message if not silent.

        Args:
            message (str): The chat message to log.
            silent (bool): Whether to suppress logging.
        """
        self._log(message, silent)

    def _initialize_chat_history(self, partner_agent: "Agent", user_input: str) -> list:
        """
        Initialize the chat history with the initial messages.

        Args:
            partner_agent (Agent): The partner agent.
            user_input (str): The initial user input.

        Returns:
            list: The initialized chat history.
        """
        # history = [{"system": partner_agent.instructions}] if hasattr(partner_agent, 'instructions') else []
        history = []
        history.append({self.name: user_input})
        return history

    def _build_context(self, sender: str,  recipient: str, message: str) -> str:
        """
        Build the initial context string.

        Args:
            sender (str): The name of the sender.
            message (str): The message content.

        Returns:
            str: The formatted context string.
        """
        return f"{message}\n"

    def _send_message(self, agent: "Agent", context: str):
        """
        Send a message to the specified agent.

        Args:
            agent (Agent): The agent to send the message to.
            context (str): The current conversation context.

        Returns:
            Any: The response from the agent.
        """
        return agent.send_request(context)

    def _process_partner_response(self, response: Any, partner_agent: "Agent", chat_history: list, context: str, silent: bool) -> Tuple[str, str, bool]:
        """
        Process the response from the partner agent.

        Args:
            response (Any): The response object from the partner agent.
            partner_agent (Agent): The partner agent.
            chat_history (list): The current chat history.
            context (str): The current conversation context.
            silent (bool): Whether to suppress logging.

        Returns:
            tuple: The latest message and the updated context.
        """
        choices = response.choices[0]
        message = choices.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            # Handle normal message
            if isinstance(message, litellm.types.utils.Message):
                content = message.get("content", "") or ""
            else:
                content = str(message) or ""

            self._add_to_history(chat_history, partner_agent.name, content)
            # print int green color
            context = self._update_context(context, partner_agent.name, content)
            self._log_separator(silent)
            self._log_message(f"\n{partner_agent.name} -> {self.name}:\n{content}\n", silent)
            self._log_separator(silent)
            return content, context, False
        else:
            # Handle tool call
            tool_call = tool_calls[0]
            return self._execute_tool_call(tool_call, partner_agent, chat_history, context, silent)

    def _execute_tool_call(self, tool_call: dict, partner_agent: "Agent", chat_history: list, context: str, silent: bool) -> Tuple[str, str]:
        """
        Execute a tool call suggested by the partner agent.

        Args:
            tool_call (dict): The tool call information.
            partner_agent (Agent): The partner agent.
            chat_history (list): The current chat history.
            context (str): The current conversation context.
            silent (bool): Whether to suppress logging.

        Returns:
            tuple: The output from the tool and the updated context.
        """
        tool_name = tool_call['function']['name']
        tool_args = json.loads(tool_call['function']['arguments'])
        
        if hasattr(self, 'functions') and tool_name in self.functions:
            output = self.functions[tool_name](**tool_args)
            context_addition = (
                f"Suggested Tool Call: {tool_name}\n"
                f"Arguments: {tool_call['function']['arguments']}\n"
            )
            self._log_separator(silent)
            self._log_message(f"\n{partner_agent.name} -> {self.name}:\n{context_addition}\n", silent)
            self._log_separator(silent)
            context = self._update_context(context, partner_agent.name, context_addition)

            self._add_to_history(chat_history, partner_agent.name, context_addition)
            return output, context, True
        else:
            error_msg = f"Function '{tool_name}' not found."
            self._log(error_msg, silent)
            context_addition = f"{partner_agent.name}: {error_msg}\n"
            context = self._update_context(context, partner_agent.name, error_msg)
            self._add_to_history(chat_history, partner_agent.name, error_msg)
            return error_msg, context, False


    def _add_to_history(self, chat_history: list, agent_name: str, message: str):
        """
        Add a message to the chat history.

        Args:
            chat_history (list): The current chat history.
            agent_name (str): The name of the agent sending the message.
            message (str): The message content.
        """
        chat_history.append({agent_name: message})

    def _update_context(self, context: str, sender: str, message: str) -> str:
        """
        Update the conversation context with a new message.

        Args:
            context (str): The current conversation context.
            sender (str): The name of the sender.
            message (str): The message content.

        Returns:
            str: The updated conversation context.
        """
        return f"\n{context}\n\n{message}\n"

    def _extract_message_content(self, response: Any) -> str:
        """
        Safely extract the 'content' from the 'message' in the response.

        Args:
            response (Any): The response object from the agent.

        Returns:
            str: The extracted message content or an empty string if unavailable.
        """
        message = response.choices[0].get("message", {})
        if isinstance(message, litellm.types.utils.Message):
            return message.get("content", "") or ""
        else:
            return str(message) or ""

    def _handle_error(self, agent_name: str, error_message: str, chat_history: list, silent: bool) -> str:
        """
        Handle errors by logging and updating the chat history.

        Args:
            agent_name (str): The name of the agent where the error occurred.
            error_message (str): The error message to log and record.
            chat_history (list): The current chat history.
            silent (bool): Whether to suppress logging.

        Returns:
            str: The error message.
        """
        self._log(error_message, silent)
        self._add_to_history(chat_history, agent_name, error_message)
        return error_message

    def _check_exit_condition(self, message: str) -> bool:
        """
        Check if the latest message is an exit command.

        Args:
            message (str): The latest message content.

        Returns:
            bool: True if the message is an exit command, False otherwise.
        """
        return isinstance(message, str) and message.lower() in ["exit", "quit"]