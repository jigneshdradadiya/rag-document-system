"""Configuration settings for RAG Document QA system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Chunking configuration
CHUNK_SIZE = 1200  # characters
CHUNK_OVERLAP = 150  # 10-15% overlap

# Retrieval configuration
TOP_K = 6  # number of chunks to retrieve

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and good quality

# LLM configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "openai" or "ollama"
LLM_TEMPERATURE = 0.0  # deterministic for QA

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")  # or gpt-3.5-turbo

# Ollama configuration (supports both local and cloud)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")  # or mistral, mixtral, etc.
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")  # For ollama.com cloud API

# Index files
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
