from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes import SQLRecordManager, index
import os
import dotenv
from chromadb.api import Settings
import chromadb

from ingestion.chromainit.chroma_collection_metadata import CollectionMetadata, get_current_time_in_iso_8601_format_utc


def get_embeddings(model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> Embeddings:
    # Running on CPU since I am using a macbook. You should change it depending on the hardware.
    model_kwargs = {'device': 'cpu'}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )


def setup_vector_store(collection_name: str, embeddings: Embeddings, persist_directory: str) -> Chroma:
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    collection_metadata = CollectionMetadata(
        data_source="SEC Filings",
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        created_by="Luke M",
        project_id="FinQuery",
        parser_version="1.1",
        created_at_iso=get_current_time_in_iso_8601_format_utc()
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        metadata=collection_metadata.model_dump(),
    )

    print(f"Successfully retrieved/created collection '{collection.name}' with metadata:")
    print(collection.metadata)

    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vector_store


def setup_record_manager(collection_name: str) -> SQLRecordManager:
    dotenv.load_dotenv()

    namespace = f"chroma/{collection_name}"

    db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

    record_manager = SQLRecordManager(
        namespace,
        db_url=db_url
    )

    record_manager.create_schema()

    return record_manager
