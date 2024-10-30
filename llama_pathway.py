import asyncio
import pathway as pw
import numpy as np
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import TokenTextSplitter
from pathway.xpacks.llm.splitters import TokenCountSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
import regex 
# Use a general HuggingFace embedding model without API keys
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

transformations_example = [
    TokenTextSplitter(
        chunk_size=150,
        chunk_overlap=10,
        separator=" ",
    ),
    embed_model,
]


data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./data",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

processing_pipeline = VectorStoreServer.from_llamaindex_components(
    *data_sources,
    transformations=transformations_example,
)

PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755

async def start_server():
    processing_pipeline.run_server(
        host=PATHWAY_HOST, port=PATHWAY_PORT, with_cache=False, threaded=False
    )

# Use asyncio to manage the event loop
if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except RuntimeError as e:
        print(f"RuntimeError occurred: {e}")
