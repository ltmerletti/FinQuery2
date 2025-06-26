from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes import SQLRecordManager, index
import os
import dotenv


def get_embeddings(model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> Embeddings:
    # Running on CPU since I am using a macbook. 
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )


def setup_vector_store(collection_name: str, embeddings: Embeddings, persist_directory: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


def setup_record_manager(collection_name: str):
    dotenv.load_dotenv()

    namespace = f"chroma/{collection_name}"

    database_url = f"postgresql+psycopg2://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}"

    record_manager = SQLRecordManager(
        namespace,
        db_url=database_url
    )

    record_manager.create_schema()

    return record_manager
