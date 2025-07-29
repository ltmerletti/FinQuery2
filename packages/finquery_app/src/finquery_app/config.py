import os
from pathlib import Path
from dotenv import load_dotenv

# --- PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

SOURCE_DATA_DIR = PROJECT_ROOT / "reports"
SOURCE_PROCESSED_DATA_DIR = PROJECT_ROOT / "reports" / "added"
CHROMA_DB_PATH = str(PROJECT_ROOT / "chromadb")
SUMMARY_DIRECTORY = PROJECT_ROOT / "reports" / "summaries"

# --- MODELS ---
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKING_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MLX_RERANKING_MODEL_NAME = "arthurcollet/Qwen3-Reranker-0.6B-mlx-6bit"

# --- Collection ---
COLLECTION_NAME = "financial_documents"

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
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY")
LMSTUDIO_MODEL_NAME = os.getenv("LMSTUDIO_MODEL_NAME")
LMSTUDIO_SMART_MODEL_NAME = os.getenv("LMSTUDIO_SMART_MODEL_NAME")
LMSTUDIO_FAST_LLM_MODEL_NAME = os.getenv("LMSTUDIO_FAST_LLM_MODEL_NAME")

# --- DB URL ---
DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Templates ---
FIND_ANSWER_FROM_RETRIEVALS_TEMPLATE = """You are an expert financial analyst AI. Your task is to provide a precise answer to the user's question based *only* on the context provided from financial documents.

Follow these steps rigorously:
1.  Carefully read the user's question to understand exactly what information is being asked for.
2.  Review each of the context snippets below. Each snippet is from a specific source document and page.
3.  Identify the single snippet that most directly and accurately answers the user's question. Ignore snippets that are only tangentially related or do not contain the specific data point requested.
4.  If no snippet contains the answer, respond with: "I could not find the answer in the provided documents."
5.  If a relevant snippet is found, construct your answer by directly extracting the information. State the fact or figure clearly and concisely.
6.  After providing the answer, you MUST cite your source in the format: "(Source: [filename], Page: [page number])".

Do not add any preamble, conversational text, or information that is not from the provided context.

---
CONTEXT SNIPPETS:
{context}
---
USER QUESTION:
{question}
---
PRECISE ANSWER:"""

FILTER_PROMPT_TEMPLATE = """You are an expert routing agent. Your goal is to determine if a user's question should be filtered to a specific type of financial document.

You will be given the user's question and a list of available document types from a database.

Analyze the user's question. If the question explicitly mentions or strongly implies one of the available document types, you must output the exact name of that document type.

Examples:
- Question: "What was the revenue from the last annual report?" and Types: ["10-K", "Press Release"] -> Output: 10-K
- Question: "What did the CEO say on the last earnings call?" and Types: ["10-K", "Earnings Call Transcript"] -> Output: Earnings Call Transcript
- Question: "What was the net income?" and Types: ["10-K", "Press Release"] -> Output: None

If the question is generic and does not point to a specific document type, you MUST output the word "None".
Do not explain your reasoning. Only output the single document type name or the word "None".

---
AVAILABLE DOCUMENT TYPES:
{document_types}
---
USER QUESTION:
{question}
---
DOCUMENT TYPE TO FILTER ON:"""

CONVERSATIONAL_QUERY_REFINER_PROMPT = """
You are a helpful and conversational financial analyst assistant. Your primary goal is to understand a user's request and, if it is ambiguous, ask clarifying questions to gather enough information to form a precise database query.

**Your Task:**
Based on the current conversation and the real-time `database_filters_summary`, you must decide on your next action. You have two possible actions: `ASK` or `FILTER`.

**1. `ASK` Action:**
If the user's request is ambiguous or missing information needed for a query, you MUST ask a clarifying question.
- Use the `database_filters_summary` to ask intelligent questions. For example, if the user asks for "Apple's revenue" and the database contains "Apple Inc." and "Apple Hospitality REIT, Inc.", you should ask which one they mean.
- If a user asks for a company or document type that is not in the summary, inform them and present the available options.
- Keep your questions concise and helpful.
- Your response for this action MUST be a JSON object with the format:
  `{{"action": "ask", "question": "Your clarifying question here."}}`

**2. `FILTER` Action:**
If you have enough information to form a precise query, generate the final search parameters.
- The `search_query` should be a concise, keyword-rich query for semantic search.
- The `metadata_filter` should contain the specific key-value pairs to filter the database. Only use keys that are present in the `database_filters_summary`.
- Your response for this action MUST be a JSON object with the format:
  `{{"action": "filter", "data": {{"search_query": "...", "metadata_filter": {{...}}}}}}`

**Available Database Filters (Real-time):**
```json
{database_filters_summary}
```

**Conversation History:**
{chat_history}

**User's Latest Message:**
{question}

**Your JSON Response:**
"""