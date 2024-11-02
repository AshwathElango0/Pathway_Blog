# config.py
import os

class Config:
    # API Keys
    LLM_API_KEY = os.getenv("LLM_API_KEY", "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc")
    
    # Paths and Hosts
    DATA_PATH = os.getenv("DATA_PATH", "../data/*")  # Path to your data files
    VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST", "0.0.0.0")
    VECTOR_STORE_PORT = int(os.getenv("VECTOR_STORE_PORT", 8000))
    TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", "/usr/local/share/tessdata")
    # Embedding and LLM Models
    EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini/gemini-1.5-flash")
