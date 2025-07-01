import json
import time

# Required for the MRI strategy
from langchain.storage import LocalFileStore
from langchain.indexes import index

# Your existing imports
from ingestion.find_file_paths import get_file_paths
from ingestion.split_into_chunks import CustomPDFLoader
from ingestion.chromainit.database_setup import setup_vector_store, setup_record_manager, get_embeddings

# Explanation because this code may seem complex:
# from the split_into_chunks.py file, we determine what the "high" value tables are using our heuristics
# we use an LLM to generate the "pointer" vector with a summary and questions about the meaning of the table
# we return two lists; this placeholder that's easier to retrieve and the data
# then we will use the file store to hold the original things
# all low-value tables and text is embedded normally, and we embed the generated embedding in lieu of the table

def main():
    # --- Step 1: Find all documents to process ---
    file_paths = get_file_paths("../reports")
    if not file_paths:
        print("No PDF files found in the '../reports' directory. Exiting.")
        return

    print("Files to be processed:")
    for file in file_paths:
        print(f" - {file.name}")

    # --- Step 2: Initialize all three storage components ---
    print("\nInitializing vector store, docstore, and record manager...")

    # The search index (for summaries, questions, text chunks)
    vector_store = setup_vector_store(
        collection_name="financial_documents",
        embeddings=get_embeddings(),
        persist_directory="../chromadb"
    )

    # The document library (for original, full tables)
    # This is a necessary component for the MultiVectorRetriever to work.
    docstore = LocalFileStore(root_path="../chromadb/docstore")

    # The record manager to avoid reprocessing files
    record_manager = setup_record_manager(collection_name="financial_documents")

    print("Initialization complete.")

    # --- Step 3: Process each file ---
    for file_path in file_paths:
        print(f"\n========================================")
        print(f"STARTING PROCESSING FOR: {file_path.name}")
        print(f"========================================")

        loader = CustomPDFLoader(str(file_path))

        # Unpack the two lists from the loader
        child_docs, representation_docs = loader.load()

        if not child_docs and not representation_docs:
            print(f"No documents were generated for {file_path.name}. Skipping.")
            continue

        # --- Handle the two document types separately ---

        # 3a. Store the original tables in the docstore (no embedding)
        if child_docs:
            print(f"Storing {len(child_docs)} original tables in the document store...")
            doc_ids = [doc.metadata["doc_id"] for doc in child_docs]

            # Serialize each Document object to a JSON string, then encode to bytes
            serialized_docs = [json.dumps(doc.dict()).encode("utf-8") for doc in child_docs]

            docstore.mset(list(zip(doc_ids, serialized_docs)))

        # 3b. Index the representations in the vector store (this creates embeddings)
        if representation_docs:
            print(f"Indexing {len(representation_docs)} representations in the vector store...")
            index(
                representation_docs,
                record_manager,
                vector_store,
                cleanup="incremental",
                source_id_key="source",
                batch_size=32
            )

        print(f"Successfully processed {file_path.name}.")

    print("\n--- Finished processing all files ---")
    final_count = vector_store._collection.count()
    print(f"The vector store collection now has {final_count} total embedded representations.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
