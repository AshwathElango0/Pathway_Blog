from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini

# Initialize the Gemini model
# gemini_llm = Gemini(model="models/gemini-ultra")

# Set the Gemini LLM in the global Settings
Settings.llm = None

# Initialize the Pathway retriever
retriever = PathwayRetriever(host="127.0.0.1", port=8755)

# Use the retriever to query data
query_engine = RetrieverQueryEngine.from_args(
    retriever,
)

response = query_engine.query("What is Pathway?")
print(str(response))
