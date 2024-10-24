from litellm import completion
import os
# set env
os.environ["GEMINI_API_KEY"] = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. San Francisco, USA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_news",
            "description": "Get the current news in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Headlines and news from a given location",
                    },
                    "unit": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What is the weather in Jaipur today?"}]

response = completion(
    model="gemini/gemini-1.5-flash",
    messages=messages,
    tools=tools,
)

# Assuming the data is stored in a variable 'response'

# Navigate through the structure to get the tool call information
tool_call = response['choices'][0]['message']['tool_calls'][0]

# Extract the 'location' and 'function name'
location = eval(tool_call['function']['arguments'])['location']  # Using eval to convert the string to a dictionary
function_name = tool_call['function']['name']

# Print the location and function name
print(f"Location: {location}")
print(f"Function Name: {function_name}")

