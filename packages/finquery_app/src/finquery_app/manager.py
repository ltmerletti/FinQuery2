import chromadb
import spacy
import tiktoken
from chromadb.api import Settings
from langchain.indexes import SQLRecordManager
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from spacy.language import Language
from tiktoken.core import Encoding

from finquery_app.config import DB_URL, LMSTUDIO_BASE_URL, LMSTUDIO_MODEL_NAME, LMSTUDIO_API_KEY, EMBEDDING_MODEL_NAME


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> Embeddings:
    # Running on CPU since I am using a macbook. You should change it depending on the hardware.
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def get_langfuse_callback():
    """
    Initializes and returns the Langfuse callback handler.
    Will not work unless dotenv has been initialized with proper env variables as per Langfuse's documentation.
    """
    return CallbackHandler()


def get_vector_store(collection_name: str, embeddings: Embeddings, persist_directory: str) -> Chroma:
    """
    Connects to an existing ChromaDB collection.
    This does NOT create or modify the collection's metadata.
    """
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    vector_store = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings, )
    return vector_store


def get_record_manager(collection_name: str) -> SQLRecordManager:
    """
    Connects to an existing SQLRecordManager namespace.
    This does NOT create the schema.
    """
    namespace = f"chroma/{collection_name}"
    record_manager = SQLRecordManager(namespace, db_url=DB_URL)
    return record_manager


def get_llm(base_url: str = None, api_key: str = None, model_name: str = None) -> ChatOpenAI:
    """
    Initializes and returns a ChatOpenAI instance.

    Defaults to using the LM Studio configuration from the config file if no
    parameters are provided. Otherwise, it uses the provided parameters.

    Args:
        base_url (str, optional): The base URL for the LLM API.
        api_key (str, optional): The API key for the LLM service.
        model_name (str, optional): The name of the model to use.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI client.
    """
    llm_base_url = base_url or LMSTUDIO_BASE_URL
    llm_api_key = api_key or LMSTUDIO_API_KEY
    llm_model_name = model_name or LMSTUDIO_MODEL_NAME

    print(f"Initializing LLM with base URL: {llm_base_url} and model: {llm_model_name}")

    llm = ChatOpenAI(model=llm_model_name, api_key=llm_api_key, base_url=llm_base_url, temperature=0.1)
    return llm


def get_spacy_model() -> Language | None:
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("FATAL ERROR: Spacy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        return None


def get_tiktoken_model() -> Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"ERROR loading tiktoken model: {e}")
        return None
