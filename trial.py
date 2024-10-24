import pathway as pw
import logging
import os
from pathway.xpacks.llm.llms import LiteLLMChat
from pathway.internals import DeferredCell

# Set up logging to debug mode
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load API Key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_LOCATOR = "gemini/gemini-1.5-flash"

# Set up the configuration for LiteLLMChat
llm_chat = LiteLLMChat(
    model=MODEL_LOCATOR,
    api_key=API_KEY,
    temperature=0.0,  # Stochasticity, keep at 0.0 for deterministic output
    max_tokens=50     # Limit on the number of tokens
)

# Sample query
query = "What is the capital of France?"

# Create a Pathway table for the query
query_table = pw.debug.table_from_markdown(f'''
query
{query}
''')

# Send a request to the LLM and get the response task
def send_request_to_gemini(query_text):
    # Prepare the message for the LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query_text}
    ]
    # Send the query to the LLM
    return llm_chat(messages)

# Async handling: Store the result in a deferred cell
def async_query_llm(query_text):
    # We create a deferred cell to manage asynchronous execution
    result_cell = DeferredCell(send_request_to_gemini, query_text)
    return result_cell

# Apply LLM query to each row in the table
result_table = query_table.select(
    result=async_query_llm(query_table.query)
)

# Trigger the Pathway execution
pw.run()

# Debugging: Print out the results from Pathway
print("Debugging result from Pathway:")
result_table.debug(name="Result Table")
