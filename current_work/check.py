import pathway as pw
# from pathway.xpacks.llm.vector_store import VectorStoreServer
from vector_store import VectorStoreServer
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
    "any/folder/of/your/choice",
    format="binary",
    mode="streaming",
    with_metadata=True,
)

splitter = SlidingWindowSplitter()
parser = ParseUnstructured() # must have libmagic in system to use this


unstructured_documents = data_sources.select(texts=parser(pw.this.data))


pw.debug.compute_and_print(unstructured_documents.select(pw.this.texts))
pw.run()