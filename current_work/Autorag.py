from typing import List
import os
import autogen
from autogen import ConversableAgent, register_function

# Define the functions that handle different types of queries

def function1(uploaded_doc_names: List[str]) -> str:
    """
    Function to perform if the given user query can be answered only by the documents uploaded by the user.
    """
    # Implement your logic here
    return f"Function1 executed with documents: {', '.join(uploaded_doc_names)}"

def function2(uploaded_doc_names: List[str]) -> str:
    """
    Function to perform if the given user query can be answered by using user-uploaded documents and external sources.
    External sources might not be mentioned but must be inferred.
    """
    # Implement your logic here
    return f"Function2 executed with documents: {', '.join(uploaded_doc_names)} and external sources."

def function3(query: str) -> str:
    """
    Function to perform if the given user query doesn't need any sources and can be answered by the agent itself.
    """
    # Implement your logic here
    return f"Function3 executed. Answering directly: {query}"

def main():
    # Load Gemini-specific configurations
    config_list_gemini = autogen.config_list_from_json(
        "config.json",
        filter_dict={
            "model": ["gemini-1.5-flash"],  # Corrected model name format
        },
    )
    
    print("Loaded Gemini Configurations:", config_list_gemini)
    
    # Define the Assistant Agent that suggests tool calls.
    assistant = ConversableAgent(
        name="Assistant",
        system_message=(
            "You are a helpful AI assistant capable of determining the appropriate method to answer user queries. "
            "Use 'Function1' if the answer relies solely on user-uploaded documents, "
            "'Function2' if it requires both user-uploaded documents and external sources, "
            "or 'Function3' if no external sources are needed. "
            "Return 'TERMINATE' when the task is done."
        ),
        llm_config={"config_list": config_list_gemini},
    )
    
    # Define the User Proxy Agent that executes tool calls.
    user_proxy = ConversableAgent(
        name="UserProxy",
        llm_config={"config_list": config_list_gemini},
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )
    
    # # Register the tool signatures with the Assistant Agent.
    # assistant.register_for_llm(
    #     name="Function1",
    #     description="Use this function when the answer relies solely on user-uploaded documents."
    # )(function1)
    
    # assistant.register_for_llm(
    #     name="Function2",
    #     description="Use this function when the answer requires both user-uploaded documents and external sources."
    # )(function2)
    
    # assistant.register_for_llm(
    #     name="Function3",
    #     description="Use this function when no external sources are needed and the answer can be provided directly by the agent."
    # )(function3)
    
    # # Register the tool functions with the User Proxy Agent.
    # user_proxy.register_for_execution(name="Function1")(function1)
    # user_proxy.register_for_execution(name="Function2")(function2)
    # user_proxy.register_for_execution(name="Function3")(function3)
    
    # Register the functions with Autogen
    register_function(
        function1,
        caller=assistant,       # The Assistant can suggest calls to Function1
        executor=user_proxy,    # The User Proxy can execute Function1
        name="Function1",
        description="Use this function when the answer relies solely on user-uploaded documents.",
    )
    
    register_function(
        function2,
        caller=assistant,       # The Assistant can suggest calls to Function2
        executor=user_proxy,    # The User Proxy can execute Function2
        name="Function2",
        description="Use this function when the answer requires both user-uploaded documents and external sources.",
    )
    
    register_function(
        function3,
        caller=assistant,       # The Assistant can suggest calls to Function3
        executor=user_proxy,    # The User Proxy can execute Function3
        name="Function3",
        description="Use this function when no external sources are needed and the answer can be provided directly by the agent.",
    )
    

    
    # Register the User Agent to send the initial message
    # Since Autogen expects interactions between two agents, we use a minimal User Agent.
    
    # Initiate the chat with a user query
    user_query = "How to solve World Hunger?"
    chat_result = user_proxy.initiate_chat(
        recipient=assistant,   # Specify the Assistant Agent as the recipient
        message=user_query,
        max_turns= 1
    )
    
    # Print the result
    print("Chat Result:", chat_result)
    
if __name__ == "__main__":
    main()
