# We will always use ChromaDB upsert rather than insert to avoid duplication
# TODO: I will still hold a mysql database with the document contents to double-check
# TODO: https://python.langchain.com/docs/how_to/indexing/ along with postgres

from typing import List
from chromadb import Collection
from ingestion.chromainit.chroma_collection_document import DocumentChunk


def add_chunks_to_collection(collection: Collection, chunks: List[DocumentChunk]):
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for i, chunk in enumerate(chunks):
        if not chunk.text.strip():
            continue

        documents_to_add.append(chunk.text)

        metadatas_to_add.append({
            'source': chunk.source_document,
            'page': chunk.page_number
        })

        chunk_id = f"{chunk.source_document}_page_{chunk.page_number}_chunk_{i}"
        ids_to_add.append(chunk_id)

    if not documents_to_add:
        print("No valid chunks to add after filtering.")
        return

    print(f"Adding {len(documents_to_add)} valid chunks to the database...")
    try:
        collection.add(
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print("Successfully added chunks to the collection.")
    except Exception as e:
        print(f"An error occurred while adding chunks: {e}")
