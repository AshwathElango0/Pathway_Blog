import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from sentence_transformers import SentenceTransformer
import asyncio
import threading
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.embedders import BaseEmbedder , SentenceTransformerEmbedder
from pathway.xpacks.llm.parsers import ParseUtf8, ParseUnstructured

model = SentenceTransformer('all-MiniLM-L6-v2')

# def embedder(text: str) -> list[float]:
#     return model.encode(text).tolist()


# def parser(data: bytes) -> list[tuple[str, dict]]:
#     text = data.decode('utf-8')
#     return [(text, {})]


# def splitter(text: str) -> list[tuple[str, dict]]:
#     chunk_size = 500
#     overlap = 50
#     chunks = []
#     for i in range(0, len(text), chunk_size - overlap):
#         chunk = text[i:i + chunk_size]
#         if chunk:
#             chunks.append((chunk, {}))
#     return chunks


data_sources = pw.io.fs.read(
    "./data",
    format="binary",
    mode="streaming",
    with_metadata=True,
)

splitter = TokenCountSplitter(min_tokens=1, max_tokens=100)
embedder = SentenceTransformerEmbedder(model ='all-MiniLM-L6-v2')
parser = ParseUnstructured() # must have libmagic in system to use this

vector_store_server = VectorStoreServer(
    data_sources,
    embedder=embedder,
    parser=parser,
    splitter=splitter,
    # doc_post_processors=[remove_extra_whitespace],  # Optional
)


PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755

def run_server():
    vector_store_server.run_server(
        host=PATHWAY_HOST,
        port=PATHWAY_PORT,
        with_cache=False,  
        threaded=False, 
    )

if __name__ == "__main__":
    run_server()
