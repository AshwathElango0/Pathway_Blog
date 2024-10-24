import pathway as pw
pw.set_license_key("demo-license-key-with-telemetry")

from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

data = pw.io.fs.read(
    "./data",
    format="binary",
    mode="streaming",
    with_metadata=True,
)