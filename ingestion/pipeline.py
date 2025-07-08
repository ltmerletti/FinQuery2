import pathlib
import shutil
import time
from typing import List

from langchain.indexes import SQLRecordManager, index
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Using absolute imports from the project root for clarity and reliability
from ingestion.chromainit.database_setup import setup_vector_store, setup_record_manager, get_embeddings
from ingestion.find_file_paths import get_file_paths
from finquery_parser.loader import CustomPDFLoader

def run_ingestion_process():
    """
    Finds, processes, and indexes all new PDF documents from the 'reports'
    directory into the vector store. This is the core, reusable logic.
    """
    # Establish paths relative to this file's location to find the project root
    project_root = pathlib.Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    chroma_dir = project_root / "chromadb"
    collection_name = "financial_documents"

    # --- Step 1: Find all documents to process ---
    file_paths = get_file_paths(str(reports_dir))
    if not file_paths:
        print("No new PDF files found in the 'reports' directory. Exiting.")
        return

    print("Files to be processed:")
    for file in file_paths:
        print(f" - {file.name}")

    # --- Step 2: Initialize database connections ONCE ---
    print("\nInitializing vector store and record manager...")
    vector_store: Chroma = setup_vector_store(
        collection_name=collection_name,
        embeddings=get_embeddings(),
        persist_directory=str(chroma_dir)
    )
    record_manager: SQLRecordManager = setup_record_manager(collection_name=collection_name)
    print("Initialization complete.")

    for file_path in file_paths:
        print(f"\n========================================")
        print(f"STARTING PROCESSING FOR: {file_path.name}")
        print(f"========================================")

        try:
            loader: CustomPDFLoader = CustomPDFLoader(str(file_path))
            docs: List[Document] = loader.load()

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
                batch_size=64
            )
            print(f"Successfully indexed {file_path.name}.")

            # Move the processed file to the 'added' subdirectory
            destination_dir = reports_dir / 'added'
            destination_dir.mkdir(exist_ok=True)
            shutil.move(file_path, destination_dir / file_path.name)
            print(f"File {file_path.name} has been moved to '{destination_dir.name}' directory.")

        except Exception as e:
            print(f"!!! An error occurred while processing {file_path.name}: {e}")
            continue

    print("\n--- Finished processing all files ---")
    final_count = vector_store._collection.count()
    print(f"The collection now has {final_count} total chunks.")
