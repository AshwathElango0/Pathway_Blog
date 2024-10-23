import pathway as pw
import logging
import json
from pathway.xpacks.llm.llms import LiteLLMChat

# Set up logging to debug mode
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# API Key and Model Locator for the Gemini Model
API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"
MODEL_LOCATOR = "gemini/gemini-1.5-flash"

# Set up the configuration for LiteLLMChat
llm_chat = LiteLLMChat(
    model=MODEL_LOCATOR,
    api_key=API_KEY,  # Make sure to pass your API key here
    temperature=0.0,  # Stochasticity, keep at 0.0 for deterministic output
    max_tokens=50     # Limit on the number of tokens
)

# Function to send a request to the LLM using LiteLLMChat
def send_request_to_gemini(query):
    # Prepare the message in the format required for LiteLLMChat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},  # System instruction
        {"role": "user", "content": query}  # User's query
    ]
    
    # Log the request
    logger.debug(f"Sending request to Gemini with messages: {json.dumps(messages, indent=2)}")
    
    try:
        # Use LiteLLMChat to send the message
        response = llm_chat(messages)
        # Log the raw response for debugging
        # logger.debug(f"Raw response from LiteLLMChat: {response}")
        print("Response object from LiteLLMChat after pw.run():")
        print(response)
        return response

    except Exception as e:
        # Log any errors that occur
        # logger.error(f"An error occurred: {e}")
        return None

# Main execution block

query = "What is the capital of France?"
response = send_request_to_gemini(query)
   
# Print the final response
if response is not None:
        print(f"Response from Gemini: {response}")
else:
        print("Failed to get a response.")

pw.run()

