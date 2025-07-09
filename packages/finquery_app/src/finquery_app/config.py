import os
from pathlib import Path
from dotenv import load_dotenv

# --- PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

SOURCE_DATA_DIR = PROJECT_ROOT / "reports"
SOURCE_PROCESSED_DATA_DIR = PROJECT_ROOT / "reports" / "added"
CHROMA_DB_PATH = str(PROJECT_ROOT / "chromadb")

# --- MODELS ---
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKING_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
LLM_NAME = "qwen3-30b-a3b-mixed-3"


# --- SECRETS ---
load_dotenv()

# --- DATABASE ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# --- LANGFUSE ---
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")

# --- OPENROUTER ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# --- LM STUDIO ---
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL")
LMSTUDIO_MODEL_NAME = os.getenv("LMSTUDIO_MODEL_NAME")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY")

# --- DB URL ---
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
