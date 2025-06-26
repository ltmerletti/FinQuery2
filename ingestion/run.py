import time
from find_file_paths import get_file_paths
from split_into_chunks import CustomPDFLoader
from chromainit.database_setup import setup_vector_store, setup_record_manager, get_embeddings
from langchain.indexes import index


def main():
    # --- Step 1: Find all documents to process ---
    file_paths = get_file_paths("../reports")
    if not file_paths:
        print("No PDF files found in the '../reports' directory. Exiting.")
        return

    print("Files to be processed:")
    for file in file_paths:
        print(f" - {file.name}")

    # --- Step 2: Initialize database connections ONCE ---
    print("\nInitializing vector store and record manager...")
    vector_store = setup_vector_store(
        collection_name="financial_documents",
        embeddings=get_embeddings(),
        persist_directory="../chromadb"
    )

    record_manager = setup_record_manager(collection_name="financial_documents")
    print("Initialization complete.")

    for file_path in file_paths:
        print(f"\n========================================")
        print(f"STARTING PROCESSING FOR: {file_path.name}")
        print(f"========================================")

        loader = CustomPDFLoader(str(file_path))

        docs = loader.load()

        if not docs:
            print(f"No documents were generated for {file_path.name}. Skipping.")
            continue

        print(f"Indexing {len(docs)} chunks for {file_path.name}...")
        index(
            docs,
            record_manager,
            vector_store,
            cleanup="incremental",
            source_id_key="source",
            batch_size=16
        )
        print(f"Successfully indexed {file_path.name}.")

    print("\n--- Finished processing all files ---")
    final_count = vector_store._collection.count()
    print(f"The collection now has {final_count} total chunks.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
