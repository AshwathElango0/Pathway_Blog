# import tempfile

# from autogen import ConversableAgent
# from autogen.coding import LocalCommandLineCodeExecutor

# # Create a temporary directory to store the code files.
# temp_dir = tempfile.TemporaryDirectory()

# # Create a local command line code executor.
# executor = LocalCommandLineCodeExecutor(
#     timeout=10,  # Timeout for each code execution in seconds.
#     work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
# )

# # Create an agent with code executor configuration.
# code_executor_agent = ConversableAgent(
#     "code_executor_agent",
#     llm_config=False,  # Turn off LLM for this agent.
#     code_execution_config={"executor": executor},  # Use the local command line code executor.
#     human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
# )


# message_with_code_block = """This is a message with code block.
# The code block is below:
# ```python
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.random.randint(0, 100, 100)
# y = np.random.randint(0, 100, 100)
# plt.scatter(x, y)
# plt.savefig('scatter.png')
# print('Scatter plot saved to scatter.png')
# ```
# This is the end of the message.
# """

# # Generate a reply for the given code.
# reply = code_executor_agent.generate_reply(messages=[{"role": "user", "content": message_with_code_block}])
# print(reply)

# import os
# print(temp_dir.name)

# ## if any file has .py at the end, print its contents
# for file in os.listdir(temp_dir.name):
#     if file.endswith(".py"):
#         with open(os.path.join(temp_dir.name, file)) as f:
#             print(f.read())

# os.open(temp_dir.name + '/scatter.png', os.O_RDONLY)
# print(os.listdir(temp_dir.name))


from agent import Agent
import litellm
# litellm.set_verbose=True
import logging
logging.disable(level=logging.ERROR)
LLM_API_KEY = 'AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0'

instructions = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply 'TERMINATE' in the end when everything is done.
"""
calculator_agent = Agent.from_model("gemini/gemini-1.5-pro", name="Assistant", instructions=instructions, api_key=LLM_API_KEY)
import regex as re
def extract_code_blocks_with_language(response):
    # Regex pattern to match code blocks, capturing the language and the code block content
    pattern = r"```(\w+)?\n([\s\S]*?)\n```"

    # Find all code blocks and their respective languages (language can be None if not specified)
    code_blocks = re.findall(pattern, response)
    
    if len(code_blocks) == 0:
        return None
    
    # return first one to run with language
    return code_blocks[0][0], code_blocks[0][1]

def match_file_to_extension(file_type : str) -> str:
    match(file_type):
        case "python":
            return ".py"
        case "shell":
            return ".sh"
        case _:
            return ".txt"

import tempfile
temp_dir = tempfile.TemporaryDirectory()
import os
import subprocess
import shutil
import signal
print(temp_dir.name)
timeout = 10
try:
    output = calculator_agent.send_request("Write Python Code to draw basic inferences from train.csv?").choices[0].message.content
    # extract the code block
    lang,code_blocks = extract_code_blocks_with_language(output)
    
    # print the code
    print(lang, '\n', code_blocks)
    # write the code to a file
    file_type = match_file_to_extension(lang)
    with open(os.path.join(temp_dir.name, f"code{file_type}"), "w") as f:
        f.write(code_blocks)
        
    # Define a handler for the timeout
    shutil.copyfile("train.csv", os.path.join(temp_dir.name, "train.csv"))
    
    def handler(signum, frame):
        raise TimeoutError("Code execution exceeded the time limit")
    # Set the signal handler and a timeout alarm
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout + 5)
    
    
    try:
        # run the code
        if lang == "python":
            result = subprocess.run(["python", os.path.join(temp_dir.name, f"code{file_type}")], capture_output=True)
            print("output =", result.stdout.decode())
        elif lang == "shell":
            result = subprocess.run(["sh", os.path.join(temp_dir.name, f"code{file_type}")], capture_output=True)
            print(result.stdout.decode())
        else:
            print("Invalid language")
    except TimeoutError as e:
        print(e)
    finally:
        # Disable the alarm
        signal.alarm(0)
    
except litellm.exceptions.BadRequestError as e:
    print(e)