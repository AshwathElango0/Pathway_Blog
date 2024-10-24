from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
# from llama_index.llms.huggingface import HuggingFaceLLM
# Initialize the Gemini model
gemini_llm = Gemini(
    model_locator="gemini/gemini-1.5-flash",
    api_key="AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc",
    temperature=0.0,
    max_tokens=100,
)
# Set the Gemini LLM in the global Settings
Settings.llm =  gemini_llm

# Initialize the Pathway retriever
retriever = PathwayRetriever(host="127.0.0.1", port=8755)

# Use the retriever to query data
query_engine = RetrieverQueryEngine.from_args(
    retriever,
)

response = query_engine.query("What happened in J and K today tell me in detail ? ")
print(str(response))
