import litellm
import json
import logging

# Set up logging to debug requests and responses
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# API Key and Model Details
API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"
MODEL_LOCATOR = "gemini/gemini-1.5-flash"  # Example of the Gemini model

# Set up the configuration for the request
config = {
    "api_key": API_KEY,  # API key to authenticate
    "model": MODEL_LOCATOR,  # Model to use
    "temperature": 0.0,  # Stochasticity of the model's output (0 is deterministic)
    "max_tokens": 50  # Limit the number of tokens in the response
}

def send_request_to_gemini(query):
    """Function to send a simple request to Gemini."""
    # Prepare the input messages in the format expected by litellm
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},  # System instruction
        {"role": "user", "content": query}  # User's query
    ]

    # Log the request for debugging purposes
    logger.debug(f"Sending request to Gemini with the following messages: {json.dumps(messages, indent=2)}")
    
    try:
        # Send the request to litellm using the `completion` function
        response = litellm.completion(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            api_key=config["api_key"]
        )
        
        # Debug: Log the raw response
        # logger.debug(f"Raw response from litellm: {json.dumps(response, indent=2)}")
        
        # Extract the content from the response
        response_content = response.choices[0]["message"]["content"]
        logger.debug(f"Extracted response content: {response_content}")
        
        return response_content

    except litellm.BadRequestError as e:
        # Handle bad requests (e.g., missing parameters or invalid inputs)
        logger.error(f"Bad request error: {e}")
    except Exception as e:
        # Handle any other exceptions
        logger.error(f"An error occurred: {e}")

# Main code to run the query
if __name__ == "__main__":
    query = "What is the capital of France?"
    response = send_request_to_gemini(query)
    
    # Print the final response
    if response:
        print(f"Response from Gemini: {response}")
    else:
        print("Failed to get a response.")
