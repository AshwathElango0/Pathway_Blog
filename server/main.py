# server.py
import os
import logging
import pathway as pw
from pathway.xpacks.llm import embedders, splitters, llms, parsers
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.udfs import DiskCache
from pathway.xpacks.llm.question_answering import AdaptiveRAGQuestionAnswerer
from customQA import HI
from config import Config
from dotenv import load_dotenv
import litellm
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
litellm.set_verbose=True

# Initialize Vector Store
def initialize_vector_store():
    logger.info("Initializing VectorStoreServer...")
    
    # Read data from the specified path
    my_folder = pw.io.fs.read(
        path=Config.DATA_PATH,       # Path to your data files
        format="binary",
        with_metadata=True
    )
    sources = [my_folder]
    
    # Define parser, splitter, and embedder
    parser = parsers.OpenParse(
    table_args={
        "parsing_algorithm": "pymupdf",
        }
    )
    text_splitter = splitters.TokenCountSplitter(max_tokens=400)
    embedder = embedders.SentenceTransformerEmbedder(model ='all-MiniLM-L6-v2')
    
    # Initialize VectorStoreServer
    vector_server = VectorStoreServer(
        *sources,
        embedder=embedder,
        splitter=text_splitter,
        parser=parser,
    )
    
    logger.info("VectorStoreServer initialized successfully.")
    return vector_server

# Initialize LLM
def initialize_llm():
    logger.info("Initializing LLM...")
    
    chat = llms.LiteLLMChat(
    model=Config.LLM_MODEL_NAME,
    api_key=Config.LLM_API_KEY,
    temperature=0.0,
    max_tokens=100,
    )
    
    logger.info("LLM initialized successfully.")
    return chat

# Initialize Adaptive RAG Question Answerer
def initialize_rag(vector_server, chat):
    logger.info("Initializing AdaptiveRAGQuestionAnswerer...")
    
    app = HI(
        indexer=vector_server,
        llm=chat,
    )
    
    logger.info("AdaptiveRAGQuestionAnswerer initialized successfully.")
    return app

# Build and run the server
def run_server():
    logger.info("Starting the RAG server...")
    
    # Initialize components
    vector_server = initialize_vector_store()
    chat = initialize_llm()
    app = initialize_rag(vector_server, chat)
    
    # Build server with host and port
    app.build_server(host=Config.VECTOR_STORE_HOST, port=Config.VECTOR_STORE_PORT)
    
    # Run the server
    app.run_server()
    logger.info("RAG server is running.")

if __name__ == "__main__":
    run_server()
