from chromainit import *
from chroma_collection_metadata import get_current_time_in_iso_8601_format_utc

collection_metadata = CollectionMetadata(
    data_source="SEC Filings",
    embedding_model_name="Qwen3-Embedding-8B",
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

print("\n--- Script Finished ---")
print("Successfully retrieved collection object:")
print(x)
print(f"\nCollection Name: {x.name}")
print(f"Collection Metadata: {x.metadata}")
