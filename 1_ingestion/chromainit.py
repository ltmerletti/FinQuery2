from chromadb.api import ClientAPI
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api.models import Collection
import chromadb
from chromadb.config import Settings


def get_embedding_function(model_name: str = "Qwen/Qwen3-Embedding-8B") -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_client() -> ClientAPI:
    client = chromadb.PersistentClient(path="../chromadb", settings=Settings(anonymized_telemetry=False))

    return client


def get_collection(client: ClientAPI, embedding_function: SentenceTransformerEmbeddingFunction, collection_name: str,
                   collection_metadata: dict) -> Collection:
    return client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata=collection_metadata,
    )
