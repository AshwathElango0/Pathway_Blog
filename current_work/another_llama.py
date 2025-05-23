import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from sentence_transformers import SentenceTransformer
import asyncio
import threading
from splitter import BaseTokenSplitter , DefaultTokenCountSplitter , SlidingWindowSplitter , SmallToBigSplitter
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.embedders import BaseEmbedder , SentenceTransformerEmbedder
from pathway.xpacks.llm.parsers import ParseUtf8, ParseUnstructured, OpenParse
import os

if 'TESSDATA_PREFIX' not in os.environ:
    os.environ['TESSDATA_PREFIX'] = '/usr/local/share/tessdata'

data_sources = pw.io.fs.read(
    "../data",
    format="binary",
    mode="streaming",
    with_metadata=True,
)

splitter = DefaultTokenCountSplitter(min_tokens=1,max_tokens=500)
embedder = SentenceTransformerEmbedder(model ='all-MiniLM-L6-v2')
parser = ParseUnstructured() # must have libmagic in system to use this

another_parser = OpenParse(
    table_args={
        "parsing_algorithm": "pymupdf",

    }
)

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
