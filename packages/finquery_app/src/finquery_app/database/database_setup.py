from pprint import pprint

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain.indexes import SQLRecordManager
from chromadb.api import Settings
import chromadb

from finquery_app.database.chroma_collection_metadata import CollectionMetadata, get_current_time_in_iso_8601_format_utc
from finquery_app.config import DB_URL, EMBEDDING_MODEL_NAME, CHROMA_DB_PATH
from finquery_app.database.manager import get_embeddings


def setup_vector_store(collection_name: str, embeddings: Embeddings, persist_directory: str) -> Chroma:
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    collection_metadata = CollectionMetadata(
        data_source="SEC Filings",
        embedding_model_name=EMBEDDING_MODEL_NAME,
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
    namespace = f"chroma/{collection_name}"


    record_manager = SQLRecordManager(
        namespace,
        db_url=DB_URL
    )

    record_manager.create_schema()

    return record_manager

def main():
    vector_store = setup_vector_store("financial_documents", get_embeddings(EMBEDDING_MODEL_NAME), CHROMA_DB_PATH)
    record_manager = setup_record_manager("financial_documents")
    pprint(vector_store)
    pprint(record_manager)

if __name__ == "__main__":
    main()