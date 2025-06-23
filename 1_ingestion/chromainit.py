from chromadb.api import ClientAPI
from chroma_collection_metadata import CollectionMetadata
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models import Collection
import chromadb
from chromadb.config import Settings


def get_parser_version() -> str:
    return "1.0"


def get_embedding_function(model_name: str = "Qwen/Qwen3-Embedding-8B") -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_client(path_to_chromadb_folder) -> ClientAPI:
    client = chromadb.PersistentClient(path=path_to_chromadb_folder, settings=Settings(anonymized_telemetry=False))

    return client


def get_collection(client: ClientAPI, embedding_function: SentenceTransformerEmbeddingFunction, collection_name: str,
                   collection_metadata: CollectionMetadata) -> Collection:
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata=collection_metadata.model_dump(),
    )
