from chromadb.api import ClientAPI
from .chroma_collection_metadata import *
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models import Collection
import chromadb
from chromadb.config import Settings


def get_parser_version() -> str:
    return "1.0"


def get_embedding_function(model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_client(path_to_chromadb_folder) -> ClientAPI:
    client = chromadb.PersistentClient(path=path_to_chromadb_folder, settings=Settings(anonymized_telemetry=False))

    return client


def get_collection(client: ClientAPI, embedding_function: SentenceTransformerEmbeddingFunction, collection_name: str,
                   collection_metadatas: CollectionMetadata) -> Collection:
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata=collection_metadatas.model_dump(),
    )


def initialize_chroma() -> Collection:
    try:
        collection_metadata = CollectionMetadata(
            data_source="SEC Filings",
            embedding_model_name="Qwen3-Embedding-0.6B",
            created_by="Luke M",
            project_id="FinQuery",
            parser_version=get_parser_version(),
            created_at_iso=get_current_time_in_iso_8601_format_utc()
        )

        x = get_collection(
            get_client("../chromadb"),
            get_embedding_function(),
            "financial_documents",
            collection_metadata
        )

        print("Successfully retrieved collection object:")
        print(x)
        print(f"\nCollection Name: {x.name}")
        print(f"Collection Metadata: {x.metadata}")
        return x
    except Exception:
        print(f"ERROR {Exception}")
        return []
